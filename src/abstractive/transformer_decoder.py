"""
Implementation of "Attention is All You Need"
Only Hierarchical Transformers modules are borrowed from https://github.com/nlpyang/hiersumm
"""

import torch
import torch.nn as nn
import numpy as np

from abstractive.attn import MultiHeadedAttention,SelfAttention
from abstractive.neural import PositionwiseFeedForward
from abstractive.transformer_encoder import PositionalEncoding


MAX_SIZE = 5000


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()



class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()


        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, layer_cache=None, step=None, para_attn=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm

        query,_ = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid,attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")

        if para_attn is not None:
            # para_attn size is batch x block_size
            # attn size is slength x batch
            batch_size = memory_bank.size(0)
            dim_per_head = self.context_attn.dim_per_head
            head_count = self.context_attn.head_count

            def shape(x):
                """  projection """
                return x.view(batch_size, -1, head_count, dim_per_head) \
                    .transpose(1, 2)

            def unshape(x):
                """  compute context """
                return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)


            if layer_cache is not None:
                value  = layer_cache['memory_values']
            else:
                value = self.context_attn.linear_values(memory_bank)
                value = shape(value)

            attn = attn * para_attn.unsqueeze(1).repeat(1,head_count,1,1) # multiply for one step
            # renormalize attention
            attn = attn / attn.sum(-1).unsqueeze(-1)
            drop_attn = self.context_attn.dropout(attn)

            mid = unshape(torch.matmul(drop_attn, value))
            mid = self.context_attn.final_linear(mid)


        output = self.feed_forward(self.drop(mid) + query)

        return output, all_input, attn
        # return output

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask




class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, device):
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, state, memory_lengths=None,
                step=None, cache=None,memory_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        n_blocks, n_tokens, batch_size, _ = memory_bank.shape
        #print(memory_bank.shape)
        memory_bank =  memory_bank.view(n_blocks*n_tokens, batch_size, -1).contiguous()
        src = state.src
        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step) 
        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        output = self.pos_emb(output, step)

        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        #print(tgt.shape)
        for i in range(self.num_layers):
            output, all_input, attn \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    1-src_pad_mask, tgt_pad_mask,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)

        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        #print(attn.shape)
        if self.training:
            attn = self._discourse_coverage_attn(attn, n_blocks)

        return outputs, state, attn


    def _discourse_coverage_attn(self, attn, n_blocks):
        """ calculate the discouse and coverage attention tensors"""
        

        #attn["attn"] = attn.transpose(1, 2).transpose(0, 1)
        # calculate discourse tensor
        b, h, t, pp = attn.shape
        n_tokens = pp//n_blocks
        attn = attn.view(b, h, t, n_blocks, n_tokens).contiguous()
        attn_dis = attn.sum(-1) # b x h x t x n_blocks
        attn_dis = attn_dis - torch.cat((torch.zeros(b,h,1,n_blocks).to(self.device), attn_dis), 2)[:,:,:t,:]
        
        # calculate coverage tensor
        atnn = attn.transpose(1,2).transpose(0,1) # t x b x h x pp
        cov = torch.zeros_like(attn[0], requires_grad=True).to(self.device)
        attn_cov = [] 
        for a in attn:
            attn_cov.append(torch.min(torch.cat((a.unsqueeze(-1), cov.unsqueeze(-1)),-1), -1)[0]) 
            cov = cov + a
        attn_cov = torch.stack(attn_cov)

        attn = {} 
        attn["dis"] = attn_dis.transpose(1,2).transpose(0,1)
        attn["cov"] = attn_cov.transpose(1,2).transpose(0,1)

        return attn
            
 

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        if(src.dim()==3):
            src = src.view(src.size(0),-1).transpose(0,1)
        else:
            src = src.transpose(0,1)
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state



class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None,
                "local_memory_keys":None,
                "local_memory_values":None,
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 1)
        if self.cache is not None:
            _recursive_map(self.cache)


class PointerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, device):
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.decoder_type = 'pointer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, state, memory_lengths=None,
                step=None, cache=None,memory_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        batch_size, n_blocks, _ = memory_bank.shape
        src = state.src
        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)
        emb =  memory_bank.index_select(0, tgt)
        emb = torch.cat((torch.zeros(batch_size, 1, self.d_model), emb), -1)
        
        assert emb.dim() == 3  # len x batch x embedding_dim

        emb = emb[:,:-1,:]


        output = emb.transpose(0, 1).contiguous()
        output = self.pos_emb(output, step)

        src_memory_bank = memory_bank       
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        #print(tgt.shape)
        for i in range(self.num_layers):
            output, all_input, attn \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    1-src_pad_mask, tgt_pad_mask,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)

        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()

        return outputs, state, attn

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        if(src.dim()==3):
            src = src.view(src.size(0),-1).transpose(0,1)
        else:
            src = src.transpose(0,1)
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state


