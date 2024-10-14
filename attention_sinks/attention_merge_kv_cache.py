"""
Adapted from https://github.com/mit-han-lab/streaming-llm
"""

from dataclasses import dataclass
import torch.nn.functional as F
import math
import torch
from typing import Callable, Tuple



def gather1d(src:torch.Tensor, dst:torch.Tensor):
    pass


def gather2d(src:torch.Tensor, dst:torch.Tensor):
    pass


def gather3d(src:torch.Tensor, dst:torch.Tensor):
    pass




DIM_TO_GATHER= {
    1: gather1d,
    2: gather2d,
    3: gather3d,
}



def pitome_text(
   metric: torch.Tensor, 
   ratio:float=1.0,
   sigma:torch.Tensor=0.5,
   n_local=256,
   n_init  = 4,
):
    with torch.no_grad():
        merged_metric = metric[..., n_init:metric.shape[1]-n_local, :]
        B,T, C = merged_metric.shape


        r = math.floor(T- T*ratio)

        merged_metric = F.normalize(merged_metric, p=2, dim=-1) 

        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        sim = merged_metric@merged_metric.transpose(-1,-2)

        energy_score = (torch.exp(-(((1 - sim)/sigma)**2 * 0.5))).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi))) 
        indices =  torch.argsort(energy_score, descending=True)

        merge_idx = indices[..., :2*r]
        protected_idx = indices[..., 2*r:]
        src_idx, dst_idx = merge_idx[..., ::2], merge_idx[..., 1::2]

        scores = sim.gather(dim=-1, index=src_idx.unsqueeze(-2).expand(B, T, r)) 
        scores = scores.gather(dim=-2, index=dst_idx.unsqueeze(-1).expand(B, r, r ))
        _, dst_idx = scores.max(dim=-1) 

         

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        B, H, T, C = x.shape
        
        x_init = x[batch_idx, : , :n_init, :].squeeze_(1)
        x_local = x[batch_idx, :, -n_local:, :].squeeze_(1)

        x = x[batch_idx, :, n_init:T-n_local, :].squeeze_(1)
        x_protected = x[batch_idx, : , protected_idx, :]
        # breakpoint()
        
        x_src, x_dst = x[batch_idx, :, src_idx, :].squeeze_(1), x[batch_idx, :, dst_idx, :].squeeze_(1)
        # breakpoint()
        x_dst = x_dst.scatter_reduce(-3, dst_idx[..., None, None].expand(B, r , H, C), x_src, reduce=mode)
         

        x = torch.cat([
            x_init,
            x_protected.transpose(1,2),
            x_dst.transpose(1,2), 
            x_local,
        ], dim=-2)
        # x = torch.cat([x_init.squeeze_(1), x_local.squeeze_(1)], dim=-2)
        return x 


    return merge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x*size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x
 


@dataclass
class AttentionMergeKVCache:
    kv_init_size: int = 4
    kv_window_size: int = 1020
    k_seq_dim: int = 2
    v_seq_dim: int = 2
    kv_type:str = 'pitome'
    kv_ratio:float = 0.6 
    kv_sigma:float = 0.25 

    def __post_init__(self):
        self.cache_size = self.kv_init_size + self.kv_window_size

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        
        # print(self.kv_ratio)
        # print(self.kv_sigma)
        # print(self.kv_init_size)
        # print(self.kv_window_size)
        merges =  [
            pitome_text(
                metric = kv[0].mean(1),
                ratio=self.kv_ratio,
                sigma=self.kv_sigma,
                n_init=self.kv_init_size,
                n_local=256,
            )
            for  i, kv in enumerate(past_key_values)
        ]
 
        return [
            [
                merge_wavg(merges[i],kv[0], None), 
                merge_wavg(merges[i],kv[1], None), 
            
            ] 
            for i, kv in enumerate(past_key_values)
        ]
        

