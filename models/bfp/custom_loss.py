import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, lambda1=0.1, lambda2=0.1, lambda3=0.1, p=2, feature_dim=256):
        super(CustomLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.p = p
        self.D = nn.Linear(feature_dim, feature_dim)  # Learnable transformation matrix
    
    def forward(self, x1, x2, z, z_prime, f_x1, f_x2):
        loss1 = F.mse_loss(self.D(x1), z)
        loss2 = F.mse_loss(self.D(x2), z_prime)
        loss3 = self.lambda1 * f_x1 + self.lambda2 * f_x2
        loss4 = self.lambda3 * torch.norm(x1 - x2, p=self.p)
        
        return loss1 + loss2 + loss3 + loss4

# Inside the observe method of Bfp:
custom_loss_fn = CustomLoss(lambda1=0.1, lambda2=0.1, lambda3=0.05, feature_dim=feats[-1].shape[1])
custom_loss = custom_loss_fn(inputs, buf_inputs, feats[-1], buf_feats_new_net[-1], torch.norm(inputs), torch.norm(buf_inputs))

loss = ce_loss + logits_distill_loss + replay_ce_loss + bfp_loss_all + custom_loss
