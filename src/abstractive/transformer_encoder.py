"""
Implementation of "Attention is All You Need"
Only Hierarchical Transformers modules are borrowed from https://github.com/nlpyang/hiersumm
"""
import math

import torch.nn as nn
import torch

from abstractive.attn import MultiHeadedAttention, MultiHeadedPooling, SelfAttention
from abstractive.neural import PositionwiseFeedForward, PositionalEncoding, sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        mask = mask.unsqueeze(1)
        context,_ = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerQueryEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerQueryEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        ## query related
        self.query_pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)
        self.query_layer_norm1 = nn.LayerNorm(d_model, eps=1e-16)
        self.query_layer_norm2 = nn.LayerNorm(d_model, eps=1e-16)

    def forward(self, inputs, query, mask, query_mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """

        b, ntokens, dim = inputs.size()
        batch_size = query.size(0)
        nblocks = int(b/batch_size)
        ## Query Operations
        query_vec = self.query_layer_norm(query)
        query_vec = self.query_pooling(query_vec, query_vec, mask_query)
        query_vec = query_vec.unsqueeze(1).unsqueeze(1)
        query_vec = query_vec.expand(batch_size, nblocks, ntokens, dim).contiguous()
        query_vec = query_vec.view(batch_size*nblocks, ntokens, dim)
        query_norm = self.query_layer_norm2(query_vec)

        input_norm = self.layer_norm(inputs)
        mask = mask.unsqueeze(1)
        context,_ = self.self_attn(input_norm, input_norm, query_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)




class TransformerPoolingLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerPoolingLayer, self).__init__()

        self.pooling_attn = MultiHeadedPooling(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        context = self.pooling_attn(inputs, inputs,
                                    mask=mask)
        out = self.dropout(context)

        return self.feed_forward(out)



class TransformerEncoderHE(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, device):
        super(TransformerEncoderHE, self).__init__()
        inter_layers = [int(i) for i in inter_layers]
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim / 2))
        self.dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList(
            [TransformerInterLayer(d_model, inter_heads, d_ff, dropout) if i in inter_layers else TransformerEncoderLayer(
                d_model, heads, d_ff, dropout)
             for i in range(num_layers)])
        self.transformer_types = ['inter' if i in inter_layers else 'local' for i in range(num_layers)]
        print(self.transformer_types)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.proj = nn.Linear(2*d_model, d_model, bias=False)

    def forward(self, src):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_blocks, n_tokens = src.size()
        # src = src.view(batch_size * n_blocks, n_tokens)
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0

        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        inter_pos_emb = self.pos_emb.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        combined_pos_emb = torch.cat([local_pos_emb, inter_pos_emb], -1)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            if (self.transformer_types[i] == 'local'):
                word_vec = self.transformer_layers[i](word_vec, word_vec,
                                                      1 - mask_local)  # all_sents * max_tokens * dim
            elif (self.transformer_types[i] == 'inter'):
                if self.transformer_types[i-1] == 'local':
                    local_vec = word_vec
                word_vec = self.transformer_layers[i](word_vec, 1 - mask_local, 1 - mask_block, batch_size, n_blocks)  # all_sents * max_tokens * dim


        global_vec = self.layer_norm(word_vec)
        local_vec = self.layer_norm(local_vec)
        mask_hier = mask_local[:, :, None].float()
        global_src_features = global_vec * mask_hier
        global_src_features = global_src_features.view(-1, global_src_features.size(-1))
        
        local_src_features = local_vec * mask_hier
        local_src_features = local_src_features.view(-1, local_src_features.size(-1))
 
        # cocat and project
        src_features = torch.cat((global_src_features, local_src_features), 1)
        src_features = self.proj(src_features)
        src_features = src_features.view(batch_size, n_blocks*n_tokens, -1)
        src_features = src_features.transpose(0,1).contiguous()

        mask = mask_local
        mask = mask.view(batch_size, n_blocks*n_tokens)
        mask = mask.unsqueeze(1)


        return src_features, mask




class TransformerEncoderOrder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, device):

        super(TransformerEncoderOrder, self).__init__() 
        inter_layers = [int(i) for i in inter_layers]
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        ######
        self.transformer_layers = nn.ModuleList(
            [TransformerInterLayer(d_model, inter_heads, d_ff, dropout) if i in inter_layers else TransformerEncoderLayer(
                d_model, heads, d_ff, dropout)
             for i in range(num_layers)])
        self.transformer_types = ['inter' if i in inter_layers else 'local' for i in range(num_layers)]
        print(self.transformer_types)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        ### self-attention based positional embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        self.pos_attn = SelfAttention(d_model, dropout)
        self.final_proj = nn.Linear(2*d_model, d_model)



    def forward(self, src):
        batch_size, n_blocks, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0
        ##### 

        combined_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                                self.embeddings.embedding_dim)

        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            if (self.transformer_types[i] == 'local'):
                word_vec = self.transformer_layers[i](word_vec, word_vec,
                                                      1 - mask_local)  # all_sents * max_tokens * dim
            elif (self.transformer_types[i] == 'inter'):
                if self.transformer_types[i-1] == 'local':
                    local_vec = word_vec
                word_vec = self.transformer_layers[i](word_vec, 1 - mask_local, 1 - mask_block, batch_size, n_blocks)  # all_sents * max_tokens * dim

        global_vec = self.layer_norm(word_vec)
        local_vec = self.layer_norm(local_vec)
        mask_hier = mask_local[:, :, None].float()
        global_src_features = global_vec * mask_hier
        global_src_features = global_src_features.view(batch_size, n_blocks * n_tokens, -1)
        global_src_features = global_src_features.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim

        mask = mask_local
        mask = mask.view(batch_size, n_blocks*n_tokens)
        mask = mask.unsqueeze(1)

        ### self attention for positional embedding
        d_model = self.d_model
        pos_features = global_src_features.transpose(0,1).contiguous()
        _, attn = self.pos_attn(pos_features, mask.squeeze(1))

        attn = attn.view(batch_size, n_blocks, n_tokens)
        para_attn = attn.sum(-1) # batch_size x n_blocks


        pe = torch.zeros(batch_size, n_blocks,d_model).cuda()
        multiplier_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model))).cuda()

        pe[:, :, 0::2] = torch.sin(para_attn.unsqueeze(-1) * multiplier_term)
        pe[:, :, 1::2] = torch.cos(para_attn.unsqueeze(-1) * multiplier_term)


        pe = pe.unsqueeze(-2).expand(batch_size, n_blocks, n_tokens, -1).contiguous()
        pe = pe.view(batch_size, n_blocks*n_tokens, -1).contiguous().transpose(0, 1)

        feats = torch.cat((global_src_features, pe), -1)

        feats = self.final_proj(feats)

        return feats, mask



class TransformerEncoderHEO(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, device):

        super(TransformerEncoderHEO, self).__init__() 
        inter_layers = [int(i) for i in inter_layers]
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        ######
        self.transformer_layers = nn.ModuleList(
            [TransformerInterLayer(d_model, inter_heads, d_ff, dropout) if i in inter_layers else TransformerEncoderLayer(
                d_model, heads, d_ff, dropout)
             for i in range(num_layers)])
        self.transformer_types = ['inter' if i in inter_layers else 'local' for i in range(num_layers)]
        print(self.transformer_types)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        ### self-attention based positional embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        self.pos_attn = SelfAttention(d_model, dropout)
        self.final_proj = nn.Linear(3*d_model, d_model)



    def forward(self, src):
        batch_size, n_blocks, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0
        ##### 

        combined_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                                self.embeddings.embedding_dim)

        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            if (self.transformer_types[i] == 'local'):
                word_vec = self.transformer_layers[i](word_vec, word_vec,
                                                      1 - mask_local)  # all_sents * max_tokens * dim
            elif (self.transformer_types[i] == 'inter'):
                if self.transformer_types[i-1] == 'local':
                    local_vec = word_vec
                word_vec = self.transformer_layers[i](word_vec, 1 - mask_local, 1 - mask_block, batch_size, n_blocks)  # all_sents * max_tokens * dim

        global_vec = self.layer_norm(word_vec)
        local_vec = self.layer_norm(local_vec)
        mask_hier = mask_local[:, :, None].float()

        global_src_features = global_vec * mask_hier
        global_src_features = global_src_features.view(batch_size, n_blocks * n_tokens, -1)
        global_src_features = global_src_features.transpose(0, 1).contiguous()

        local_src_features = local_vec * mask_hier
        local_src_features = local_src_features.view(batch_size, n_blocks*n_tokens, -1)
        local_src_features = local_src_features.transpose(0,1).contiguous()

        mask = mask_local
        mask = mask.view(batch_size, n_blocks*n_tokens)
        mask = mask.unsqueeze(1)

 
        ### self attention for positional embedding
        d_model = self.d_model
        pos_features = global_src_features.transpose(0,1).contiguous()
        _, attn = self.pos_attn(pos_features, mask.squeeze(1))

        attn = attn.view(batch_size, n_blocks, n_tokens)
        para_attn = attn.sum(-1) # batch_size x n_blocks


        if self.device=="cuda":
            pe = torch.zeros(batch_size, n_blocks,d_model).cuda()
            multiplier_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model))).cuda()

        else:
            pe = torch.zeros(batch_size, n_blocks,d_model)
            multiplier_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))



        pe[:, :, 0::2] = torch.sin(para_attn.unsqueeze(-1) * multiplier_term)
        pe[:, :, 1::2] = torch.cos(para_attn.unsqueeze(-1) * multiplier_term)


        pe = pe.unsqueeze(-2).expand(batch_size, n_blocks, n_tokens, -1).contiguous()
        pe = pe.view(batch_size, n_blocks*n_tokens, -1).contiguous().transpose(0, 1)

        feats = torch.cat((global_src_features, local_src_features, pe), -1)

        feats = self.final_proj(feats)

        return feats, mask



class TransformerEncoderQuery(nn.Module):
    
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, num_query_layers, device):
        super(TransformerEncoderQuery, self).__init__()
        
        inter_layers = [int(i) for i in inter_layers]
        para_layers = [i for i in range(num_layers)]
        for i in inter_layers:
            para_layers.remove(i)
        query_layer = para_layers[-1]
        para_layers = para_layers[:-1]
        self.device = device
        self.num_query_layers = num_query_layers
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim / 2))
        self.dropout = nn.Dropout(dropout)

        ### Query Encoder
        self.transformer_query_encoder = nn.ModuleList([TransformerEncoderLayer(d_model, heads, d_ff, dropout) for i in range(num_query_layers)])
        self.query_pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim, buffer_name='qpe')
        ######
        self.transformer_layers = nn.ModuleList(
            [TransformerInterLayer(d_model, inter_heads, d_ff, dropout) if i in inter_layers else TransformerEncoderLayer(
                d_model, heads, d_ff, dropout) if i in para_layers else TransformerQueryEncoderLayer(d_model, heads, d_ff, dropout)
             for i in range(num_layers)])
        self.transformer_types = ['inter' if i in inter_layers else 'para_local' if i in para_layers else 'query_local' for i in range(num_layers)]
        print(self.transformer_types)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)



    def forward(self, src, query, query_details=False):
        batch_size, n_blocks, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_query = 1 - query.data.eq(padding_idx)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0

        ### operations for query encoding

        _, qn_tokens = query.size()
        qpos_emb = self.query_pos_emb.qpe[:,:qn_tokens].expand(batch_size, qn_tokens, self.embeddings.embedding_dim)
        qemb = self.embeddings(query) * math.sqrt(self.embeddings.embedding_dim) + qpos_emb 
        query_vec = self.query_pos_emb.dropout(qemb)
        for i in range(self.num_query_layers):
            query_vec = self.transformer_query_encoder[i](query_vec, query_vec, 1-mask_query)

        ##### 
        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                      int(self.embeddings.embedding_dim / 2))
        inter_pos_emb = self.pos_emb.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                      int(self.embeddings.embedding_dim / 2))
        combined_pos_emb = torch.cat([local_pos_emb, inter_pos_emb], -1)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            if (self.transformer_types[i] == 'para_local'):
                word_vec = self.transformer_layers[i](word_vec, word_vec,
                                                      1 - mask_local)  # all_sents * max_tokens * dim
            elif self.transformer_layers[i] == 'query_local':
                word_vec = self.transformer_layers[i](word_vec, query_vec, 1-mask_local, 1-mask_query)

            elif (self.transformer_types[i] == 'inter'):
                if 'local' in self.transformer_types[i-1]:
                    local_vec = word_vec
                word_vec = self.transformer_layers[i](word_vec, 1 - mask_local, 1 - mask_block, batch_size, n_blocks)  # all_sents * max_tokens * dim

        global_vec = self.layer_norm(word_vec)
        local_vec = self.layer_norm(local_vec)
        mask_hier = mask_local[:, :, None].float()
        global_src_features = global_vec * mask_hier
        global_src_features = global_src_features.view(batch_size, n_blocks * n_tokens, -1)
        global_src_features = global_src_features.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim 

        mask = mask_local
        mask = mask.view(batch_size, n_blocks*n_tokens)
        mask = mask.unsqueeze(1)
        if query_details:
            return global_src_features, mask, query_vec, mask_query.unsqueeze(1)
        else:
            return global_src_features, mask

class TransformerEncoderHEQ(nn.Module):
    
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, device):
        super(TransformerEncoderHEQ, self).__init__()
        
        inter_layers = [int(i) for i in inter_layers]
        para_layers = [i for i in range(num_layers)]
        for i in inter_layers:
            para_layers.remove(i)
        query_layer = para_layers[-1]
        para_layers = para_layers[:-1]
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim / 2))
        self.dropout = nn.Dropout(dropout)

        ### Query Encoder
        self.transformer_query_encoder = TransformerEncoderLayer(d_model, heads, d_ff, dropout)
        self.query_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.query_pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim, buffer_name='qpe')
        ######
        self.transformer_layers = nn.ModuleList(
            [TransformerInterLayer(d_model, inter_heads, d_ff, dropout) if i in inter_layers else TransformerEncoderLayer(
                d_model, heads, d_ff, dropout) if i in para_layers else TransformerQueryEncoderLayer(d_model, heads, d_ff, dropout)
             for i in range(num_layers)])
        self.transformer_types = ['inter' if i in inter_layers else 'para_local' if i in para_layers else 'query_local' for i in range(num_layers)]
        print(self.transformer_types)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.proj = nn.Linear(2*d_model, d_model, bias=False) # for concating local and global layer



    def forward(self, src, query):
        batch_size, n_blocks, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_query = 1 - query.data.eq(padding_idx)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0

        ### operations for query encoding

        _, qn_tokens = query.size()
        qpos_emb = self.query_pos_emb.qpe[:,:qn_tokens].expand(batch_size, qn_tokens, self.embeddings.embedding_dim)
        qemb = self.embeddings(query) * math.sqrt(self.embeddings.embedding_dim) + qpos_emb 
        qemb = self.query_pos_emb.dropout(qemb)
        query_vec = self.transformer_query_encoder(qemb, qemb, 1-mask_query)

        ##### 
        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                      int(self.embeddings.embedding_dim / 2))
        inter_pos_emb = self.pos_emb.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                      int(self.embeddings.embedding_dim / 2))
        combined_pos_emb = torch.cat([local_pos_emb, inter_pos_emb], -1)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            if (self.transformer_types[i] == 'para_local'):
                word_vec = self.transformer_layers[i](word_vec, word_vec,
                                                      1 - mask_local)  # all_sents * max_tokens * dim
            elif self.transformer_layers[i] == 'query_local':
                word_vec = self.transformer_layers[i](word_vec, query_vec, 1-mask_local, 1-mask_query)

            elif (self.transformer_types[i] == 'inter'):
                if 'local' in self.transformer_types[i-1]:
                    local_vec = word_vec
                word_vec = self.transformer_layers[i](word_vec, 1 - mask_local, 1 - mask_block, batch_size, n_blocks)  # all_sents * max_tokens * dim

        
        global_vec = self.layer_norm(word_vec)
        local_vec = self.layer_norm(local_vec)
        mask_hier = mask_local[:, :, None].float()
        global_src_features = global_vec * mask_hier
        global_src_features = global_src_features.view(-1, global_src_features.size(-1))
        
        local_src_features = local_vec * mask_hier
        local_src_features = local_src_features.view(-1, local_src_features.size(-1))
 
        # cocat and project
        src_features = torch.cat((global_src_features, local_src_features), 1)
        src_features = self.proj(src_features)
        src_features = src_features.view(batch_size, n_blocks*n_tokens, -1)
        src_features = src_features.transpose(0,1).contiguous()

        mask = mask_local
        mask = mask.view(batch_size, n_blocks*n_tokens)
        mask = mask.unsqueeze(1)
        return src_features, mask




class TransformerEncoderHERO(nn.Module):
    
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, device):
        super(TransformerEncoderHERO, self).__init__()
        
        inter_layers = [int(i) for i in inter_layers]
        para_layers = [i for i in range(num_layers)]
        for i in inter_layers:
            para_layers.remove(i)
        query_layer = para_layers[-1]
        para_layers = para_layers[:-1]
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim))
        self.dropout = nn.Dropout(dropout)

        ### Query Encoder
        self.transformer_query_encoder = TransformerEncoderLayer(d_model, heads, d_ff, dropout)
        self.query_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.query_pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim, buffer_name='qpe')
        ######
        self.transformer_layers = nn.ModuleList(
            [TransformerInterLayer(d_model, inter_heads, d_ff, dropout) if i in inter_layers else TransformerEncoderLayer(
                d_model, heads, d_ff, dropout) if i in para_layers else TransformerQueryEncoderLayer(d_model, heads, d_ff, dropout)
             for i in range(num_layers)])
        self.transformer_types = ['inter' if i in inter_layers else 'para_local' if i in para_layers else 'query_local' for i in range(num_layers)]
        print(self.transformer_types)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        ### self-attention based positional embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        self.pos_attn = SelfAttention(d_model, dropout)
        self.final_proj = nn.Linear(3*d_model, d_model)



    def forward(self, src, query):
        batch_size, n_blocks, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_query = 1 - query.data.eq(padding_idx)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0

        ### operations for query encoding

        _, qn_tokens = query.size()
        qpos_emb = self.query_pos_emb.qpe[:,:qn_tokens].expand(batch_size, qn_tokens, self.embeddings.embedding_dim)
        qemb = self.embeddings(query) * math.sqrt(self.embeddings.embedding_dim) + qpos_emb 
        qemb = self.query_pos_emb.dropout(qemb)
        query_vec = self.transformer_query_encoder(qemb, qemb, 1-mask_query)

        ##### 
        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                      int(self.embeddings.embedding_dim))
        combined_pos_emb = local_pos_emb
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            if (self.transformer_types[i] == 'para_local'):
                word_vec = self.transformer_layers[i](word_vec, word_vec,
                                                      1 - mask_local)  # all_sents * max_tokens * dim
            elif self.transformer_layers[i] == 'query_local':
                word_vec = self.transformer_layers[i](word_vec, query_vec, 1-mask_local, 1-mask_query)

            elif (self.transformer_types[i] == 'inter'):
                if 'local' in self.transformer_types[i-1]:
                    local_vec = word_vec
                word_vec = self.transformer_layers[i](word_vec, 1 - mask_local, 1 - mask_block, batch_size, n_blocks)  # all_sents * max_tokens * dim

        
       
        global_vec = self.layer_norm(word_vec)
        local_vec = self.layer_norm(local_vec)
        mask_hier = mask_local[:, :, None].float()

        global_src_features = global_vec * mask_hier
        global_src_features = global_src_features.view(batch_size, n_blocks * n_tokens, -1)
        global_src_features = global_src_features.transpose(0, 1).contiguous()

        local_src_features = local_vec * mask_hier
        local_src_features = local_src_features.view(batch_size, n_blocks * n_tokens, -1)
        local_src_features = local_src_features.transpose(0, 1).contiguous()



        mask = mask_local
        mask = mask.view(batch_size, n_blocks*n_tokens)
        mask = mask.unsqueeze(1)

 
        ### self attention for positional embedding
        d_model = self.d_model
        pos_features = global_src_features.transpose(0,1).contiguous()
        _, attn = self.pos_attn(pos_features, mask.squeeze(1))

        attn = attn.view(batch_size, n_blocks, n_tokens)
        para_attn = attn.sum(-1) # batch_size x n_blocks

        pe = torch.zeros(batch_size, n_blocks,d_model).cuda()
        multiplier_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model))).cuda()

        pe[:, :, 0::2] = torch.sin(para_attn.unsqueeze(-1) * multiplier_term)
        pe[:, :, 1::2] = torch.cos(para_attn.unsqueeze(-1) * multiplier_term)


        pe = pe.unsqueeze(-2).expand(batch_size, n_blocks, n_tokens, -1).contiguous()
        pe = pe.view(batch_size, n_blocks*n_tokens, -1).contiguous().transpose(0, 1)

        feats = torch.cat((global_src_features, local_src_features, pe), -1)

        feats = self.final_proj(feats)

        return feats, mask



class TransformerInterEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, device):
        super(TransformerInterEncoder, self).__init__()
        inter_layers = [int(i) for i in inter_layers]
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim / 2))
        self.dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList(
            [TransformerInterLayer(d_model, inter_heads, d_ff, dropout) if i in inter_layers else TransformerEncoderLayer(
                d_model, heads, d_ff, dropout)
             for i in range(num_layers)])
        self.transformer_types = ['inter' if i in inter_layers else 'local' for i in range(num_layers)]
        print(self.transformer_types)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_blocks, n_tokens = src.size()
        # src = src.view(batch_size * n_blocks, n_tokens)
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0

        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        inter_pos_emb = self.pos_emb.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        combined_pos_emb = torch.cat([local_pos_emb, inter_pos_emb], -1)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            if (self.transformer_types[i] == 'local'):
                word_vec = self.transformer_layers[i](word_vec, word_vec,
                                                      1 - mask_local)  # all_sents * max_tokens * dim
            elif (self.transformer_types[i] == 'inter'):
                word_vec = self.transformer_layers[i](word_vec, 1 - mask_local, 1 - mask_block, batch_size, n_blocks)  # all_sents * max_tokens * dim


        word_vec = self.layer_norm(word_vec)
        mask_hier = mask_local[:, :, None].float()
        src_features = word_vec * mask_hier
        src_features = src_features.view(batch_size, n_blocks * n_tokens, -1)
        src_features = src_features.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim
        mask_hier = mask_hier.view(batch_size, n_blocks * n_tokens, -1)
        mask_hier = mask_hier.transpose(0, 1).contiguous()

        unpadded = [torch.masked_select(src_features[:, i], mask_hier[:, i].byte()).view([-1, src_features.size(-1)])
                    for i in range(src_features.size(1))]
        max_l = max([p.size(0) for p in unpadded])
        mask_hier = sequence_mask(torch.tensor([p.size(0) for p in unpadded]), max_l).to(self.device)
        mask_hier = 1 - mask_hier[:, None, :]

        unpadded = torch.stack(
            [torch.cat([p, torch.zeros(max_l - p.size(0), src_features.size(-1)).to(self.device)]) for p in unpadded], 1)
        #return unpadded, mask_hier
        mask = mask_local
        mask = mask.view(batch_size, n_blocks*n_tokens)
        mask = mask.unsqueeze(1)

        return src_features, mask


class TransformerInterLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerInterLayer, self).__init__()
        self.d_model, self.heads = d_model, heads
        self.d_per_head = self.d_model // self.heads
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)

        self.layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)

        self.inter_att = MultiHeadedAttention(1, self.d_per_head, dropout, use_final_linear=False)

        self.linear = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask_local, mask_inter, batch_size, n_blocks):
        word_vec = self.layer_norm1(inputs)
        mask_inter = mask_inter.unsqueeze(1).expand(batch_size, self.heads, n_blocks).contiguous()
        mask_inter = mask_inter.view(batch_size * self.heads, 1, n_blocks)

        # block_vec = self.pooling(word_vec, mask_local)

        block_vec = self.pooling(word_vec, word_vec, mask_local)
        block_vec = block_vec.view(-1, self.d_per_head)
        block_vec = self.layer_norm2(block_vec)
        block_vec = block_vec.view(batch_size, n_blocks, self.heads, self.d_per_head)
        block_vec = block_vec.transpose(1, 2).contiguous().view(batch_size * self.heads, n_blocks, self.d_per_head)

        block_vec,_ = self.inter_att(block_vec, block_vec, block_vec, mask_inter)  # all_sents * max_tokens * dim
        block_vec = block_vec.view(batch_size, self.heads, n_blocks, self.d_per_head)
        block_vec = block_vec.transpose(1, 2).contiguous().view(batch_size * n_blocks, self.heads * self.d_per_head)
        block_vec = self.linear(block_vec)

        block_vec = self.dropout(block_vec)
        block_vec = block_vec.view(batch_size * n_blocks, 1, -1)
        out = self.feed_forward(inputs + block_vec)

        return out



class TransformerQueryInterLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerQueryInterLayer, self).__init__()
        self.d_model, self.heads = d_model, heads
        self.d_per_head = self.d_model // self.heads
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)

        ### query related
        self.paragraph_pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)
        self.query_pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)
        self.query_layer_norm1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.query_layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)

        self.layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)

        self.inter_att = MultiHeadedAttention(1, self.d_per_head, dropout, use_final_linear=False)

        self.linear = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, query_inputs, mask_local, mask_inter, mask_query, batch_size, n_blocks):

        ### QUERY OPERATIONS
        query_vec = self.query_layer_norm1(query_inputs)
        query_vec = self.query_pooling(query_vec, query_vec, mask_query)
        query_vec = query_vec.view(-1, self.d_per_head)
        query_vec = self.query_layer_norm2(query_vec)
        query_vec = query_vec.view(batch_size, 1, self.heads, self.d_per_head)
        query_vec = query_vec.expand(batch_size, n_blocks, self.heads, self.d_per_head)
        query_vec = query_vec.transpose(1, 2).contiguous().view(batch_size * self.heads, n_blocks, self.d_per_head)

        ######

        word_vec = self.layer_norm1(inputs)
        mask_inter = mask_inter.unsqueeze(1).expand(batch_size, self.heads, n_blocks).contiguous()
        mask_inter = mask_inter.view(batch_size * self.heads, 1, n_blocks)
        block_vec = self.paragraph_pooling(word_vec, word_vec, mask_local)
        block_vec = block_vec.view(-1, self.d_per_head)
        block_vec = self.layer_norm2(block_vec)
        block_vec = block_vec.view(batch_size, n_blocks, self.heads, self.d_per_head)
        block_vec = block_vec.transpose(1, 2).contiguous().view(batch_size * self.heads, n_blocks, self.d_per_head)
        # MODIFIED THE BELOW ONE
        block_vec,_ = self.inter_att(block_vec, block_vec, query_vec, mask_inter)  # all_sents * max_tokens * dim
        block_vec = block_vec.view(batch_size, self.heads, n_blocks, self.d_per_head)
        block_vec = block_vec.transpose(1, 2).contiguous().view(batch_size * n_blocks, self.heads * self.d_per_head)
        block_vec = self.linear(block_vec)

        block_vec = self.dropout(block_vec)
        block_vec = block_vec.view(batch_size * n_blocks, 1, -1)
        out = self.feed_forward(inputs + block_vec)

        return out


class TransformerNewInterLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerNewInterLayer, self).__init__()

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout)

        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.inter_att = MultiHeadedAttention(heads, d_model, dropout, use_final_linear=True)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask_local, mask_inter, batch_size, n_blocks):
        word_vec = self.layer_norm1(inputs)
        mask_inter = mask_inter.unsqueeze(1)
        # block_vec = self.pooling(word_vec, mask_local)

        block_vec = self.pooling(word_vec, word_vec, mask_local)
        _mask_local = ((1 - mask_local).unsqueeze(-1)).float()
        block_vec_avg = torch.sum(word_vec * _mask_local, 1) / (torch.sum(_mask_local, 1) + 1e-9)
        block_vec = self.dropout(block_vec) + block_vec_avg
        block_vec = self.layer_norm2(block_vec)
        block_vec = block_vec.view(batch_size, n_blocks, -1)
        block_vec = self.inter_att(block_vec, block_vec, block_vec, mask_inter)  # all_sents * max_tokens * dim
        block_vec = self.dropout(block_vec)
        block_vec = block_vec.view(batch_size * n_blocks, 1, -1)
        out = self.feed_forward(inputs + block_vec)

        return out



class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        self.transformer_local = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_hier = 1 - src.data.eq(padding_idx)
        out = self.pos_emb(emb)

        for i in range(self.num_layers):
            out = self.transformer_local[i](out, out, 1 - mask_hier)  # all_sents * max_tokens * dim
        out = self.layer_norm(out)

        mask_hier = mask_hier[:, :, None].float()
        src_features = out * mask_hier
        src_features = src_features.transpose(0, 1).contiguous()
        mask_hier = mask_hier.transpose(0, 1).contiguous()
        return src_features, mask_hier
