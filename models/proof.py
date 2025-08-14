import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import  Proof_Net
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
from utils.data_manager import LaionData
import math
import matplotlib.pyplot as plt
import os


num_workers = 2
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args=args

        self._train_transformer=False
        self._network = Proof_Net(args, False)
        
        self.batch_size = get_attribute(args,"batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        
        self._known_classes = 0
        self.use_cos = get_attribute(args, "use_cos", False)
        # Model Fusion additions
        self.task_models = []  # Store a deepcopy of the model after each task
        self.task_prototypes = {}  # Store prototypes for each task
        self.unified_prototypes = None  # Store the fused prototypes
        self.unified_model = None  # Store the fused model
        # Safe DataLoader workers (Colab/CPU friendly)
        try:
            device_type = getattr(self._device, "type", "cpu")
        except Exception:
            device_type = "cpu"
        self.loader_workers = get_attribute(args, "num_workers", 0 if device_type == "cpu" else 2)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
    
    def cal_prototype(self, trainloader, model):
        model.eval()
        # Limit samples per class to speed up on CPU/Colab
        max_per_class = get_attribute(self.args, "prototype_max_per_class", 32)
        class_list = list(range(self._known_classes, self._total_classes))
        counts = {int(c): 0 for c in class_list}
        sums = {}
        feature_dim = model.feature_dim if hasattr(model, "feature_dim") else None
        if feature_dim is not None:
            for c in class_list:
                sums[int(c)] = torch.zeros(feature_dim, dtype=torch.float32, device="cpu")

        total_needed = len(class_list) * max_per_class
        collected = 0
        log_every = 20
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                _, data, label = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.convnet.encode_image(data, True)  # [bs, dim]
                embedding = embedding.detach().cpu()
                label = label.detach().cpu()

                for emb, lab in zip(embedding, label):
                    lab_i = int(lab.item())
                    if lab_i in counts and counts[lab_i] < max_per_class:
                        if lab_i not in sums:
                            # lazy init if feature_dim unknown
                            sums[lab_i] = emb.clone()
                        else:
                            sums[lab_i] += emb
                        counts[lab_i] += 1
                        collected += 1
                if i % log_every == 0:
                    done_classes = sum(1 for c in class_list if counts[c] >= max_per_class)
                    logging.info(f"[Fusion] Prototype sampling progress: batches={i}, collected={collected}/{total_needed}, classes_done={done_classes}/{len(class_list)}")
                # Early stop when enough samples collected for all classes
                if all(counts[c] >= max_per_class for c in class_list):
                    break

        # Compute mean prototype per class from sums/counts; fallback to previous if none collected
        for class_index in class_list:
            if counts[class_index] > 0:
                proto = (sums[class_index] / counts[class_index]).to(self._device)
                self._network.img_prototypes[class_index] = proto
            else:
                # keep existing prototype (initialized as zeros) or leave unchanged
                pass

        logging.info("[Fusion] Prototypes computed for task {} (max_per_class={})".format(self._cur_task, max_per_class))
        # Save prototypes for this task
        self.task_prototypes[self._cur_task] = self._network.img_prototypes.clone().detach().cpu()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        self._network.update_prototype(self._total_classes)
        self._network.update_context_prompt() # add context prompts

        self._network.extend_task()
        
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        logging.info("[Fusion] Preparing train dataset...")
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self._network.to(self._device)
        
        # Configure DataLoaders with safe workers and pin_memory
        pin_mem = getattr(self._device, "type", "cpu") == "cuda"
        logging.info(f"[Fusion] Creating DataLoaders (workers={self.loader_workers}, pin_memory={pin_mem})...")
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.loader_workers,
            pin_memory=pin_mem,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.loader_workers,
            pin_memory=pin_mem,
        )

     #   train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", )
     #   self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        logging.info("[Fusion] Preparing prototype dataset and loader...")
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.loader_workers,
            pin_memory=pin_mem,
        )

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        logging.info("[Fusion] Computing prototypes...")
        self.cal_prototype(self.train_loader_for_protonet, self._network)
        logging.info("[Fusion] Starting projection training...")
        self._train_proj(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        # Model Fusion: Save a deepcopy of the model after each task
        import copy
        self.task_models.append(copy.deepcopy(self._network).cpu())
        # Model Fusion: Fuse models after each task
        logging.info("[Fusion] Fusing models...")
        self.fuse_models()
    
    def _train_proj(self, train_loader, test_loader, train_loader_for_protonet):
        self._train_transformer=True
        self._network.to(self._device)
        
        # Save checkpoint before training
        checkpoint_path = f"checkpoint_task_{self._cur_task}_before_training.pth"
        torch.save({
            'task': self._cur_task,
            'model_state_dict': self._network.state_dict(),
            'prototypes': self._network.img_prototypes,
        }, checkpoint_path)
        logging.info(f"[Fusion] Checkpoint saved to {checkpoint_path}")
       
        for name, param in self._network.convnet.named_parameters():
            if 'logit_scale' not in name:
                param.requires_grad = False
        self._network.freeze_projection_weight_new()
        
        if self.args['optimizer']=='sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer']=='adam': 
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)

        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        prog_bar = tqdm(range(self.tuned_epoch))
        cliploss = ClipLoss()

        total_labels = class_to_label[:self._total_classes] # mask all known classes
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            logging.info(f"[Fusion] Starting epoch {epoch + 1}/{self.tuned_epoch}")
            for i, (_, inputs, targets) in enumerate(train_loader):
                if i % 10 == 0:  # Log every 10 batches
                    logging.info(f"[Fusion] Epoch {epoch + 1}, Batch {i}/{len(train_loader)}")
                labels = [class_to_label[y] for y in targets]
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                
                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                text_features = self._network.encode_text(texts) # [total_classes, dim]
                text_feas = text_features / text_features.norm(dim=-1, keepdim=True)
                image_features = self._network.encode_image(inputs)
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True) #[bs, dim]
                image_features, text_features, logit_scale, proto_feas=self._network.forward_transformer(img_feas, text_feas,self._train_transformer)
                logits = image_features@text_features.T # [bs, allclasses]

                texts=[templates.format(inst) for inst in labels]
                clip_text_feas=self._network.encode_text(self._network.tokenizer(texts).to(self._device))#total,dim
                clip_text_norm=clip_text_feas.norm(dim=-1, keepdim=True)
                clip_text_feas = clip_text_feas / clip_text_norm

                clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)

                loss = F.cross_entropy(logits, targets)

                protoloss = F.cross_entropy(image_features @ proto_feas.T, targets)

                total_loss = loss+clip_loss+protoloss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                losses += total_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            logging.info(f"[Fusion] Epoch {epoch + 1} completed - Train Acc: {train_acc:.2f}%")
            logging.info("[Fusion] Computing test accuracy...")
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task,epoch + 1,self.args['tuned_epoch'],losses / len(train_loader),train_acc, test_acc,  )
            prog_bar.set_description(info)
            logging.info(f"[Fusion] Epoch {epoch + 1} summary: {info}")


    def _compute_accuracy(self, model, loader):
        # Use unified model for inference if available
        eval_model = self.unified_model if self.unified_model is not None else self._network
        eval_model.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes] # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = eval_model.tokenizer(texts).to(self._device)
                class_embeddings = eval_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features=eval_model.encode_image(inputs)
                transf_image_features, transf_text_features, _, proto_feas = eval_model.forward_transformer(image_features, text_features,self._train_transformer)
                outputs = transf_image_features @ transf_text_features.T
                proto_outputs= transf_image_features @ proto_feas.T
                original_outputs= image_features @ text_features.T
                outputs = original_outputs+outputs+proto_outputs
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


    def _eval_cnn(self, loader):
        # Use unified model for inference if available
        eval_model = self.unified_model if self.unified_model is not None else self._network
        eval_model.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes] # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = eval_model.tokenizer(texts).to(self._device)
                class_embeddings = eval_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features=eval_model.encode_image(inputs)
                transf_image_features, transf_text_features, _, proto_feas = eval_model.forward_transformer(image_features, text_features,self._train_transformer)
                outputs = transf_image_features @ transf_text_features.T
                proto_outputs= transf_image_features @ proto_feas.T
                original_outputs= image_features @ text_features.T
                outputs = original_outputs+outputs+proto_outputs
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def fuse_models(self):
        """
        Fuse all task prototypes into a unified prototype set (mean of all prototypes for each class).
        Optionally, can also fuse projection layers or context prompts if needed.
        """
        if not self.task_prototypes:
            return
        # Find the max number of classes across all tasks
        max_classes = max([p.shape[0] for p in self.task_prototypes.values()])
        # Stack all prototypes (pad if needed)
        proto_list = []
        for t, proto in self.task_prototypes.items():
            if proto.shape[0] < max_classes:
                pad = torch.zeros(max_classes - proto.shape[0], proto.shape[1])
                proto = torch.cat([proto, pad], dim=0)
            proto_list.append(proto)
        # Average prototypes across tasks
        unified_proto = torch.stack(proto_list, dim=0).mean(dim=0)
        self.unified_prototypes = unified_proto
        # Optionally, create a unified model (here, just update img_prototypes)
        if self.unified_model is None:
            import copy
            self.unified_model = copy.deepcopy(self._network).to(self._device)
        self.unified_model.img_prototypes = self.unified_prototypes.clone().to(self._device)
        # Log
        logging.info(f"[Model Fusion] Unified prototypes shape: {self.unified_prototypes.shape}")


