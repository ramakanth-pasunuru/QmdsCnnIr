""" Multi-Head Attention module 
    Majority of code borrowed from https://github.com/nlpyang/hiersumm
"""
import math
import torch
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if(self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type in ["context", "global_context", "local_context"]:
                query = self.linear_query(query)
                if layer_cache is not None:
                    if type in ["context", "global_context"]:
                        mkeys_str = "memory_keys"
                        mvalues_str = "memory_values"
                    elif type in ["local_context"]:
                        mkeys_str = "local_memory_keys"
                        mvalues_str = "local_memory_values"

                    if layer_cache[mkeys_str] is None:
                        key, value = self.linear_keys(key),\
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache[mkeys_str],\
                                   layer_cache[mvalues_str]
                    layer_cache[mkeys_str] = key
                    layer_cache[mvalues_str] = value
                else:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
 
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)


        drop_attn = self.dropout(attn)
        if(self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output, attn
        else:
            context = torch.matmul(drop_attn, value)
            return context, attn





class MultiHeadedPooling(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super(MultiHeadedPooling, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim,
                                     head_count)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if (use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)
        self.use_final_linear = use_final_linear

    def forward(self, key, value, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x, dim=dim_per_head):
            """  projection """
            return x.view(batch_size, -1, head_count, dim) \
                .transpose(1, 2)

        def unshape(x, dim=dim_per_head):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim)

        scores = self.linear_keys(key)
        value = self.linear_values(value)

        scores = shape(scores, 1).squeeze(-1)
        value = shape(value)


        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.sum((drop_attn.unsqueeze(-1) * value), -2)
        if (self.use_final_linear):
            context = unshape(context).squeeze(1)
            output = self.final_linear(context)
            return output
        else:
            return context


class SelfAttention(nn.Module):

    def __init__(self, model_dim, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.Va = nn.Linear(model_dim, 1, bias=False)
        self.Wa = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, t, n = x.size()
        
        proj = torch.tanh(self.Wa(x.view(b*t, n).contiguous()))
        scores = self.Va(proj)
        scores = scores.view(b,t).contiguous()

        if mask is not None:
            scores = scores.masked_fill(mask, -1e18)

        attn = torch.softmax(scores, -1)
        drop_attn = self.dropout(attn)

        context = torch.sum((drop_attn.unsqueeze(-1)*x), -2)

        return context, attn

 
