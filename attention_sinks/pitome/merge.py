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
      

      init_metric = metric[..., :init_size, :]
      local_metric = metric[..., -window_size:, :]
      merged_metric = metric[..., init_size:T-window_size, :]
      r = math.floor(T- T*ratio)

      merged_metric = F.normalize(merged_metric, p=2, dim=-1) 

      batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
      sim = merged_metric@merged_metric.transpose(-1,-2)

      sigma = 1 - margin 
      isolation_score = (torch.exp(-(((1 - sim)/sigma)**2 * 0.5))).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi))) 
      indices =  torch.argsort(isolation_score, descending=True)

      merge_idx = indices[..., :2*r]
      protected_idx = indices[..., 2*r:]
      a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2]

      scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
      scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r ))
      _, dst_idx = scores.max(dim=-1) 

   def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
      if class_token:
         x_cls=x[:,0,:].unsqueeze(1)
         x=x[:,1:,:]
      else:
         x_cls = None

      B, T, C = x.shape
      protected = x[batch_idx, protected_idx, :]
      src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
      dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)

      if x_cls is not None:
         return torch.cat([x_cls, protected, dst], dim=1)
      else:
         return torch.cat([protected, dst], dim=1)

   isolation_score = 1 - F.softmax(isolation_score, dim=-1) 

   if class_token:
      return merge, torch.cat([torch.ones(B, 1).to(metric.device), isolation_score], dim=-1)[..., None]
   return merge, isolation_score[..., None] 
