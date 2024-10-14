import torch
import math
import torch.nn.functional as F

def pitome_text(
   metric: torch.Tensor, 
   ratio:float=1.0,
   margin:torch.Tensor=0.5,
   class_token: bool = False,
   window_size=256,
   init_size = 4
):
   with torch.no_grad():
      B,T,C = metric.shape
      merged_metric = metric[..., init_size:T-window_size, :]

      r = math.floor(T- T*ratio)

      merged_metric = F.normalize(merged_metric, p=2, dim=-1) 

      batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
      sim = merged_metric@merged_metric.transpose(-1,-2)

      sigma = 1 - margin 
      energy_score = (torch.exp(-(((1 - sim)/sigma)**2 * 0.5))).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi))) 
      indices =  torch.argsort(energy_score, descending=True)

      merge_idx = indices[..., :2*r]
      protected_idx = indices[..., 2*r:]
      src_idx, dst_idx = merge_idx[..., ::2], merge_idx[..., 1::2]

      scores = sim.gather(dim=-1, index=src_idx.unsqueeze(-2).expand(B, T, r)) 
      scores = scores.gather(dim=-2, index=dst_idx.unsqueeze(-1).expand(B, r, r ))
      _, dst_idx = scores.max(dim=-1) 

         

   def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
      B,T,C = x.shape
      x_init = x[batch_idx, :init_size, :]
      x_local = x[batch_idx, -window_size:, :]

      x = x[batch_idx, init_size:T-window_size, :]
      x_protected = x[batch_idx, protected_idx, :]
      x_src, x_dst = x[batch_idx, src_idx, :], x[batch_idx,  dst_idx, :]
      x_dst = x_dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), x_src, reduce=mode)

      sorted_idx = torch.sort(torch.cat([protected_idx, dst_idx]),dim=1, descending=False).indices
      x_merged = torch.gather(torch.cat([x_protected, x_dst], dim=1), dim=1, index=sorted_idx)
      

      return torch.cat([x_init, x_merged, x_local], dim=1)


   return merge
