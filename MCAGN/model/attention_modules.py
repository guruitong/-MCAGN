import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Dict, Optional, Tuple, Union,  Callable, List
from model.utils import shift_dim

class SelfAttention(nn.Module):
    def __init__(self, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_dropout = attn_dropout

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        _, _, *shape, _ = q.shape

        # flatten to b, h, (d1, ..., dn), dim_q/dim_kv
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out, attn_probs = scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            head_mask=head_mask,
            attn_dropout=self.attn_dropout if self.training else 0.0,
        )

        return out.unflatten(2, shape), attn_probs


class GraphMultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        n_head: int,
        attn_module: nn.Module = SelfAttention(),
        add_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim_q % n_head != 0 or dim_kv % n_head != 0:
            raise ValueError(
                "The hidden size of q, k, v must be a multiple of the number of attention heads."
            )

        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.n_head = n_head
        self.query = nn.Linear(dim_q, dim_q, bias=add_bias)  # q
        self.key = nn.Linear(dim_kv, dim_q, bias=add_bias)  # k
        self.value = nn.Linear(dim_kv, dim_q, bias=add_bias)  # v
        self.output = nn.Linear(dim_q, dim_q, bias=True)  # c

        self.attn = attn_module

        self.cache: Optional[Dict[str, Tensor]] = None

    def forward(
        self,
        q: Tensor,
        kv: Optional[Tensor] = None,
        return_attn_weights: bool = False,
        use_cache: bool = False,
        causal: bool = False,
        **attn_kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        #if isinstance(self.attn, AxialAttention) and causal:
            #raise TypeError("Causal axial attention is not supported.")

        # If kv is specified use those inputs for cross-attention, otherwise use q
        k = v = q if kv is None else kv
        # compute q
        q = split_multihead(self.query(q), self.n_head)

        # For causal k, v are provided step-wise so we should always compute them
        # For non-causal skip computing k, v if they have been cached
        if causal or not self.cache:
            k = split_multihead(self.key(k), self.n_head)
            v = split_multihead(self.value(v), self.n_head)

        # fast decoding by caching past key, value tensors
        if use_cache:
            if not self.cache:
                # initialize the cache with the present k, v
                self.cache = dict(k=k.clone(), v=v.clone())
            else:
                if causal:
                    # append present k, v to past k, v
                    # for autoregressive decoding inputs are flattened as 1D sequences
                    # so are the cached tensors: (b, n_heads, seq_len, c)
                    k_, v_ = self.cache["k"], self.cache["v"]
                    self.cache["k"] = torch.cat([k_, k], dim=2)
                    self.cache["v"] = torch.cat([v_, v], dim=2)
                # override the present k, v with the cache
                k, v = self.cache["k"], self.cache["v"]

        attn_out = self.attn(q, k, v, **attn_kwargs)
        attn_probs = None
        # Unpack if attn module also returns attn probs
        if isinstance(attn_out, tuple):
            attn_out, attn_probs = attn_out
        a = merge_multihead(attn_out)
        a = self.output(a)

        if return_attn_weights:
            return a, attn_probs
        else:
            return a


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Union[int, List[int]]] = None,
        dropout: float = 0.5,
        activation: Callable[..., nn.Module] = nn.ReLU,
        normalization: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        layers = nn.ModuleList()

        if hidden_dims is None:
            hidden_dims = []

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization:
                layers.append(normalization(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Optional[Tensor] = None,
    head_mask: Optional[Tensor] = None,
    attn_dropout: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    # Take the dot product between "query" and "key" and scale to get the raw attention scores.
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / torch.sqrt(torch.tensor(q.shape[-1]))
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor with the computed attention weights
    # at the positions we want to attend and -inf for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    if attention_mask is not None:
        attn = attn.masked_fill(attention_mask == 0, float("-inf"))
    # Normalize the attention scores to probabilities
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)  # b, h, d1, ..., q_dn, k_dn
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn = F.dropout(attn, p=attn_dropout)
    # Mask heads if we want to
    if head_mask is not None:
        attn = attn * head_mask
    # For each query sum over the key/value dim with attention weights
    a = torch.matmul(attn, v)  # b, h, d1, ..., q_dn, c

    return a, attn


def split_multihead(x: Tensor, n_head: int) -> Tensor:
    x = x.unflatten(-1, (n_head, -1))
    # Rearrange to put head dim first, (b, n_head, d1, ..., dn, c // n_head)
    x = shift_dim(x, -2, 1)
    return x

def merge_multihead(x: Tensor) -> Tensor:
    return shift_dim(x, 1, -2).flatten(start_dim=-2)