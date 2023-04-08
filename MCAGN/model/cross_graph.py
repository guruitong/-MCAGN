from typing import Callable, List, NamedTuple, Optional, Tuple, Union
from torch import nn, Tensor
from model.attention_modules import GraphMultiHeadAttention, SelfAttention, MLP
from model.normalizations import Fp32LayerNorm

class CrossGraph(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        layer_norm_eps: float = 1e-12,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        # attention block
        self.attention = GraphMultiHeadAttention(
            dim_q=d_model,
            dim_kv=d_model,
            n_head=n_head,
            attn_module=SelfAttention(dropout),
        )
        self.attention_dropout = nn.Dropout(dropout)
        # cross attention block
        self.cross_attention = GraphMultiHeadAttention(
            dim_q=d_model,
            dim_kv=d_model,
            n_head=n_head,
            attn_module=SelfAttention(dropout),
        )
        self.cross_attention_dropout = nn.Dropout(dropout)
        # feedforward block
        self.feedforward = MLP(
            d_model, d_model, dim_feedforward, dropout=dropout, activation=activation
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        # layernorms
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _self_attention_block(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        output = self.attention(
            hidden_states, attention_mask=attention_mask, return_attn_weights=False
        )
        output = self.attention_dropout(output)
        return output

    def _cross_attention_block(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.cross_attention(
            hidden_states,
            encoder_hidden_states,
            attention_mask=cross_attention_mask,
            return_attn_weights=False,
        )
        output = self.cross_attention_dropout(output)
        return output

    def _feedforward_block(self, hidden_states: Tensor) -> Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = hidden_states
        kv = encoder_hidden_states
        inputs = self.attention_layernorm(x)
        attn_output = self._self_attention_block(inputs, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_norm_output = self.cross_attention_layernorm(attn_residual)
        cross_attention_output = self._cross_attention_block(
            attn_norm_output, kv, cross_attention_mask
        )
        cross_attention_residual = cross_attention_output + attn_norm_output
        cross_attention_norm_output = self.feedforward_layernorm(
            cross_attention_residual
        )
        ff_residual = cross_attention_norm_output + self._feedforward_block(
            cross_attention_norm_output
        )
        return ff_residual

    def _forward_postnorm(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = hidden_states
        kv = encoder_hidden_states
        attn_output = self._self_attention_block(x, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_norm_output = self.attention_layernorm(attn_residual)
        cross_attention_output = self._cross_attention_block(
            attn_norm_output, kv, cross_attention_mask
        )
        cross_attention_residual = cross_attention_output + attn_norm_output
        cross_attention_norm_output = self.cross_attention_layernorm(
            cross_attention_residual
        )
        ff_residual = cross_attention_norm_output + self._feedforward_block(
            cross_attention_norm_output
        )
        outputs = self.feedforward_layernorm(ff_residual)
        return outputs

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.norm_first:
            return self._forward_prenorm(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
            )
        else:
            return self._forward_postnorm(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
            )