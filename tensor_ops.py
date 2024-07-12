# tensor_ops.py
import torch
import torch.nn.functional as F
import opt_einsum

def einsum(equation, *operands):
    return opt_einsum.contract(equation, *operands)

def softmax(x, dim=-1):
    return F.softmax(x, dim=dim)

def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return weight * (x - mean) / (var + eps).sqrt() + bias

def apply_rotary_emb(x, freqs_cis):
    x_r, x_i = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_out = torch.view_as_complex(torch.stack([x_r, x_i], dim=-1))
    freqs_cis = freqs_cis.to(x_out.device)
    return torch.view_as_real(x_out * freqs_cis).flatten(3)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
