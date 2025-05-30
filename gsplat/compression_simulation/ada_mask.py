import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AnnealingMask(nn.Module):
    def __init__(self, input_shape, device, total_iters=30_000, 
                 start_temp=5.0, end_temp=0.1, 
                 annealing_start_iter=10_000, target_sparsity=0.2):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.zeros(input_shape, device=device)+1)
        self.total_iters = total_iters
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.annealing_start_iter = annealing_start_iter
        self.current_iter = 0
        self.target_sparsity = target_sparsity
        
    def get_temperature(self, current_step):
        if current_step < self.annealing_start_iter:
            return self.start_temp
        
        progress = (current_step - self.annealing_start_iter) / \
                  (self.total_iters - self.annealing_start_iter)
        progress = min(max(progress, 0), 1) 
        
        temperature = self.start_temp * math.exp(
            math.log(self.end_temp / self.start_temp) * progress
        )
        return temperature
    
    def forward(self, x, current_step):
        if self.training:
            self.current_iter = current_step
            temperature = self.get_temperature(current_step)
            mask = torch.sigmoid(self.mask_logits / temperature)
        else:
            mask = (torch.sigmoid(self.mask_logits) >= 0.5).float()
        return x * mask
    
    @torch.no_grad()
    def get_binary_mask(self):
        return (torch.sigmoid(self.mask_logits) >= 0.5).float()
    
    def get_sparsity_loss(self, lambda_l1=0.01, lambda_target=0.1):
        temperature = self.get_temperature(self.current_iter)
        mask = torch.sigmoid(self.mask_logits / temperature)
        
        l1_loss = lambda_l1 * torch.mean(mask)
        
        current_sparsity = torch.mean(mask)
        
        target = torch.tensor(self.target_sparsity).to(mask.device)
        kl_loss = lambda_target * F.binary_cross_entropy(current_sparsity, target)
        
        return l1_loss + kl_loss
    
    @torch.no_grad()
    def get_mask_ratio(self):
        return torch.sum((torch.sigmoid(self.mask_logits) >= 0.5)) / self.mask_logits.shape[0]
    
