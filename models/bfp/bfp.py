'''# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import torch
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import functional as F

from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from backbone.MNISTMLP import MNISTMLP
from utils.args import *
from utils.lowrank_reg import LowRankReg
from utils.routines import forward_loader_all_layers

from .args import set_best_args
from .utils import *
from .projector_manager import ProjectorManager

class Bfp(ContinualModel):
    NAME = 'bfp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        set_best_args(args)

        # Then call the super constructor
        super(Bfp, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.buffer = Buffer(self.args.buffer_size, self.device, class_balance = self.args.class_balance)

        # if resnet_skip_relu, modify the backbone to skip relu at the end of each major block
        if self.args.resnet_skip_relu:
            self.net.skip_relu(last=self.args.final_feat)

        # For domain-IL MNIST datasets, we should use the logits from the buffer
        if self.args.dataset in ['perm-mnist', 'rot-mnist']:
            self.args.use_buf_logits = True

        # if the old net is not used, then set the old_only and use_buf_logits flags
        if self.args.no_old_net:
            self.args.old_only = True
            self.args.use_buf_logits = True

        assert not (self.args.new_only and self.args.old_only)

        # initialize the projectors used for BFP
        self.projector_manager = ProjectorManager(self.args, self.net.net_channels, self.device)

    def begin_task(self, dataset, t=0, start_epoch=0):
        super().begin_task(dataset, t, start_epoch)
        self.projector_manager.begin_task(dataset, t, start_epoch)

    def observe(self, inputs, labels, not_aug_inputs):
        # Regular CE loss on the online data
        outputs, feats = self.net.forward_all_layers(inputs)
        ce_loss = self.loss(outputs, labels)

        def sample_buffer_and_forward(transform = self.transform):
            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=transform)
            buf_inputs, buf_labels, buf_logits, buf_task_labels = buf_data[0], buf_data[1], buf_data[2], buf_data[3]
            buf_feats = [buf_data[4]] if self.args.use_buf_feats else None
            buf_logits_new_net, buf_feats_new_net = self.net.forward_all_layers(buf_inputs)

            return buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net

        logits_distill_loss = 0.0
        replay_ce_loss = 0.0
        bfp_loss_all = 0.0
        bfp_loss_dict = None

        if not self.buffer.is_empty():
            
            if self.args.alpha_distill > 0:
                if self.args.no_resample and "buf_inputs" in locals(): pass # No need to resample
                else: buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = sample_buffer_and_forward()

                if (not self.args.use_buf_logits) and (self.old_net is not None):
                    with torch.no_grad():
                        buf_logits = self.old_net(buf_inputs)
                        
                logits_distill_loss = self.args.alpha_distill * F.mse_loss(buf_logits_new_net, buf_logits)

           
            if self.args.alpha_ce > 0:
                if self.args.no_resample and "buf_inputs" in locals(): pass # No need to resample
                else: buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = sample_buffer_and_forward()
                
                replay_ce_loss = self.args.alpha_ce * self.loss(buf_logits_new_net, buf_labels)

            
            if self.old_net is not None and self.projector_manager.bfp_flag:
                if not self.args.new_only:
                    buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = sample_buffer_and_forward()

                if self.args.use_buf_feats:
                    # new and old features should be both a list 
                    # And in this case, we only care about the last layer
                    feats_comb = buf_feats_new_net[-1:]
                    feats_old = buf_feats
                    mask_new = torch.ones_like(buf_labels).bool()
                    mask_old = torch.zeros_like(buf_labels).bool()
                else:
                    # Inputs, feats and labels for the online and buffer data, concatenated
                    if self.args.new_only:
                        inputs_comb = inputs
                        labels_comb = labels
                        feats_comb = feats
                    elif self.args.old_only:
                        mask_old = buf_labels < self.task_id * self.args.N_CLASSES_PER_TASK
                        inputs_comb = buf_inputs[mask_old]
                        labels_comb = buf_labels[mask_old]
                        feats_comb = [f[mask_old] for f in  buf_feats_new_net]
                    else:
                        inputs_comb = torch.cat((inputs, buf_inputs), dim=0)
                        labels_comb = torch.cat((labels, buf_labels), dim=0)
                        feats_comb = [torch.cat((f, bf), dim=0) for f, bf in zip(feats, buf_feats_new_net)]

                    mask_old = labels_comb < self.task_id * self.args.N_CLASSES_PER_TASK
                    mask_new = labels_comb >= self.task_id * self.args.N_CLASSES_PER_TASK

                    # Forward data through the old network to get the old features
                    with torch.no_grad():
                        self.old_net.eval()
                        _, feats_old = self.old_net.forward_all_layers(inputs_comb)
                
                bfp_loss_all, bfp_loss_dict = self.projector_manager.compute_loss(
                    feats_comb, feats_old, mask_new, mask_old)
                
        loss = ce_loss + logits_distill_loss + replay_ce_loss + bfp_loss_all

        self.opt.zero_grad()
        self.projector_manager.before_backward()

        loss.backward()
        
        self.opt.step()
        self.projector_manager.step()

        task_labels = torch.ones_like(labels) * self.task_id
        if self.args.use_buf_feats:
            # Store the unpooled version of the final-layer features in the buffer
            final_feats = feats[-1]
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                logits=outputs.data, 
                                task_labels=task_labels,
                                final_feats=final_feats.data)
        else:
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                logits=outputs.data, 
                                task_labels=task_labels)

        log_dict = {
            "train/loss": loss, 
            "train/ce_loss": ce_loss, 
            "train/logits_distill_loss": logits_distill_loss,
            "train/replay_ce_loss": replay_ce_loss,
            "train/bfp_loss_all": bfp_loss_all,
        }
        if bfp_loss_dict is not None:
            for k, v in bfp_loss_dict.items(): log_dict.update({"train/" + k: v})
        wandb.log(log_dict)

        return loss.item()

    def end_task(self, dataset):
        self.old_net = copy.deepcopy(self.net)
        self.old_net.eval()

        self.projector_manager.end_task(dataset, self.old_net)'''


# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from backbone.MNISTMLP import MNISTMLP
from utils.args import *
from utils.lowrank_reg import LowRankReg
from utils.routines import forward_loader_all_layers
from .args import set_best_args
from .utils import *
from .projector_manager import ProjectorManager

class Bfp(ContinualModel):
    NAME = 'bfp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        # Add missing hyperparameters with default values
        args.alpha_kd = getattr(args, 'alpha_kd', 1.0)
        args.alpha_feature = getattr(args, 'alpha_feature', 0.5)
        args.alpha_attention = getattr(args, 'alpha_attention', 0.1)
        args.alpha_replay = getattr(args, 'alpha_replay', 1.0)
        args.augment_replay = getattr(args, 'augment_replay', True)
        
        set_best_args(args)
        super(Bfp, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.buffer = Buffer(self.args.buffer_size, self.device, class_balance=self.args.class_balance)

        if self.args.resnet_skip_relu:
            self.net.skip_relu(last=self.args.final_feat)

        if self.args.dataset in ['perm-mnist', 'rot-mnist']:
            self.args.use_buf_logits = True

        if self.args.no_old_net:
            self.args.old_only = True
            self.args.use_buf_logits = True

        assert not (self.args.new_only and self.args.old_only)

        self.projector_manager = ProjectorManager(self.args, self.net.net_channels, self.device)
        
        # Loss components
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def begin_task(self, dataset, t=0, start_epoch=0):
        super().begin_task(dataset, t, start_epoch)
        self.projector_manager.begin_task(dataset, t, start_epoch)

    def compute_improved_loss(self, outputs, labels, buf_data=None, old_outputs=None, old_feats=None):
        ce_loss = self.loss(outputs, labels)
        
        kd_loss = 0.0
        feature_loss = 0.0
        attention_loss = 0.0
        replay_loss = 0.0
        
        # Knowledge distillation
        if self.old_net is not None and old_outputs is not None:
            temperature = 2.0
            soft_targets = F.softmax(old_outputs / temperature, dim=1)
            soft_outputs = F.log_softmax(outputs / temperature, dim=1)
            kd_loss = self.kl_div(soft_outputs, soft_targets) * (temperature ** 2)
            
        # Feature similarity
        if old_feats is not None:
            for new_f, old_f in zip(outputs.feats, old_feats):
                feature_loss += 1 - self.cos_sim(new_f, old_f.detach()).mean()
        
        # Attention preservation
        if hasattr(self.net, 'get_attention_maps') and old_feats is not None:
            new_attention = self.net.get_attention_maps(outputs.feats[-1])
            old_attention = self.old_net.get_attention_maps(old_feats[-1])
            attention_loss = F.mse_loss(new_attention, old_attention.detach())
        
        # Replay loss
        if buf_data is not None:
            buf_inputs, buf_labels = buf_data[0], buf_data[1]
            buf_outputs = self.net(buf_inputs)
            replay_ce = self.loss(buf_outputs, buf_labels)
            
            if self.args.augment_replay:
                #aug_buf_inputs = self.transform(buf_inputs)
                aug_buf_inputs = torch.stack([self.transform(img).to(self.device) for img in buf_inputs])
                aug_buf_outputs = self.net(aug_buf_inputs)
                replay_loss = replay_ce + F.mse_loss(buf_outputs, aug_buf_outputs) * 0.1
            else:
                replay_loss = replay_ce
        
        total_loss = (
            ce_loss +
            self.args.alpha_kd * kd_loss +
            self.args.alpha_feature * feature_loss +
            self.args.alpha_attention * attention_loss +
            self.args.alpha_replay * replay_loss
        )
        
        return total_loss, {
            'ce_loss': ce_loss,
            'kd_loss': kd_loss,
            'feature_loss': feature_loss,
            'attention_loss': attention_loss,
            'replay_loss': replay_loss
        }

    def observe(self, inputs, labels, not_aug_inputs):
        outputs, feats = self.net.forward_all_layers(inputs)
        
        old_outputs, old_feats = None, None
        if self.old_net is not None:
            with torch.no_grad():
                old_outputs, old_feats = self.old_net.forward_all_layers(inputs)
        
        buf_data = None
        if not self.buffer.is_empty():
            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
        
        loss, loss_dict = self.compute_improved_loss(
            outputs=outputs,
            labels=labels,
            buf_data=buf_data,
            old_outputs=old_outputs,
            old_feats=old_feats
        )
        
        if self.old_net is not None and self.projector_manager.bfp_flag:
            if not self.args.new_only:
                buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = \
                    self.sample_buffer_and_forward()
            
            if self.args.use_buf_feats:
                feats_comb = buf_feats_new_net[-1:]
                feats_old = buf_feats
                mask_new = torch.ones_like(buf_labels).bool()
                mask_old = torch.zeros_like(buf_labels).bool()
            else:
                if self.args.new_only:
                    inputs_comb = inputs
                    labels_comb = labels
                    feats_comb = feats
                elif self.args.old_only:
                    mask_old = buf_labels < self.task_id * self.args.N_CLASSES_PER_TASK
                    inputs_comb = buf_inputs[mask_old]
                    labels_comb = buf_labels[mask_old]
                    feats_comb = [f[mask_old] for f in buf_feats_new_net]
                else:
                    inputs_comb = torch.cat((inputs, buf_inputs), dim=0)
                    labels_comb = torch.cat((labels, buf_labels), dim=0)
                    feats_comb = [torch.cat((f, bf), dim=0) for f, bf in zip(feats, buf_feats_new_net)]

                mask_old = labels_comb < self.task_id * self.args.N_CLASSES_PER_TASK
                mask_new = labels_comb >= self.task_id * self.args.N_CLASSES_PER_TASK

                with torch.no_grad():
                    self.old_net.eval()
                    _, feats_old = self.old_net.forward_all_layers(inputs_comb)
            
            bfp_loss_all, bfp_loss_dict = self.projector_manager.compute_loss(
                feats_comb, feats_old, mask_new, mask_old)
            loss += bfp_loss_all
            loss_dict.update(bfp_loss_dict)
        
        self.opt.zero_grad()
        self.projector_manager.before_backward()
        loss.backward()
        self.opt.step()
        self.projector_manager.step()
        
        task_labels = torch.ones_like(labels) * self.task_id
        if self.args.use_buf_feats:
            final_feats = feats[-1]
            self.buffer.add_data(
                examples=not_aug_inputs,
                labels=labels,
                logits=outputs.data,
                task_labels=task_labels,
                final_feats=final_feats.data
            )
        else:
            self.buffer.add_data(
                examples=not_aug_inputs,
                labels=labels,
                logits=outputs.data,
                task_labels=task_labels
            )
        
        log_dict = {"train/loss": loss}
        log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
        wandb.log(log_dict)

        return loss.item()

    def sample_buffer_and_forward(self, transform=None):
        if transform is None:
            transform = self.transform
            
        buf_data = self.buffer.get_data(self.args.minibatch_size, transform=transform)
        buf_inputs, buf_labels, buf_logits, buf_task_labels = buf_data[0], buf_data[1], buf_data[2], buf_data[3]
        buf_feats = [buf_data[4]] if self.args.use_buf_feats else None
        
        if (not self.args.use_buf_logits) and (self.old_net is not None):
            with torch.no_grad():
                buf_logits = self.old_net(buf_inputs)
                
        buf_logits_new_net, buf_feats_new_net = self.net.forward_all_layers(buf_inputs)
        
        return buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net

    def end_task(self, dataset):
        self.old_net = copy.deepcopy(self.net)
        self.old_net.eval()
        self.projector_manager.end_task(dataset, self.old_net)