"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
Majority of code borrowed from https://github.com/nlpyang/hiersumm
"""

from abstractive.optimizer import Optimizer
from abstractive.transformer_encoder import TransformerEncoder, TransformerInterEncoder,  \
                                            TransformerEncoderHE, TransformerEncoderOrder,\
                                            TransformerEncoderQuery, TransformerEncoderHEQ, \
                                            TransformerEncoderHEO, TransformerEncoderHERO
from abstractive.transformer_decoder import TransformerDecoder



import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch

def build_optim(args, model, checkpoint):
    """ Build optimizer """
    optim = Optimizer(
        args.optim, args.lr, args.max_grad_norm,
        beta1=args.beta1, beta2=args.beta2,
        decay_method=args.decay_method,
        warmup_steps=args.warmup_steps, model_size=args.enc_hidden_size)
    
    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '' and not args.fine_tune:
        optim.optimizer.load_state_dict(checkpoint['optim'])
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    
    return optim


def get_generator(dec_hidden_size, vocab_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class Summarizer(nn.Module):
    def __init__(self,args, word_padding_idx, vocab_size, device, checkpoint=None):
        self.args = args
        super(Summarizer, self).__init__()
        # self.spm = spm
        self.vocab_size = vocab_size
        self.device = device
        # src_dict = fields["src"].vocab
        # tgt_dict = fields["tgt"].vocab

        src_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
        tgt_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)

        if (self.args.share_embeddings):
            tgt_embeddings.weight = src_embeddings.weight

        if self.args.model_type == 'hier':
            if (self.args.hier):
                self.encoder = TransformerInterEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                       self.args.ff_size, self.args.enc_dropout, src_embeddings, inter_layers=self.args.inter_layers, inter_heads= self.args.inter_heads, device=device)
            else:
                self.encoder = TransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                  self.args.ff_size,
                                                  self.args.enc_dropout, src_embeddings)

        
        elif self.args.model_type == 'he':
            self.encoder = TransformerEncoderHE(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                       self.args.ff_size, self.args.enc_dropout, src_embeddings, 
                                                       inter_layers=self.args.inter_layers, inter_heads= self.args.inter_heads, 
                                                       device=device)
        
        elif self.args.model_type == 'order':
            self.encoder = TransformerEncoderOrder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                       self.args.ff_size, self.args.enc_dropout, src_embeddings, 
                                                       inter_layers=self.args.inter_layers, inter_heads= self.args.inter_heads, 
                                                       device=device)


        elif self.args.model_type == 'query':
            self.encoder = TransformerEncoderQuery(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                       self.args.ff_size, self.args.enc_dropout, src_embeddings, 
                                                       inter_layers=self.args.inter_layers, inter_heads= self.args.inter_heads, 
                                                       num_query_layers=self.args.query_layers, device=device)


        elif self.args.model_type == 'heq':
            self.encoder = TransformerEncoderHEQ(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                       self.args.ff_size, self.args.enc_dropout, src_embeddings, 
                                                       inter_layers=self.args.inter_layers, inter_heads= self.args.inter_heads, 
                                                       device=device)

        elif self.args.model_type == 'heo':
            self.encoder = TransformerEncoderHEO(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                       self.args.ff_size, self.args.enc_dropout, src_embeddings, 
                                                       inter_layers=self.args.inter_layers, inter_heads= self.args.inter_heads, 
                                                       device=device)


        elif self.args.model_type == 'hero':
            self.encoder = TransformerEncoderHERO(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                       self.args.ff_size, self.args.enc_dropout, src_embeddings, 
                                                       inter_layers=self.args.inter_layers, inter_heads= self.args.inter_heads, 
                                                       device=device)

            


        self.decoder = TransformerDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.heads,
                d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings, device=device)
        


        self.generator = get_generator(self.args.dec_hidden_size, self.vocab_size, device)
        if self.args.share_decoder_embeddings:
            self.generator[0].weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            # checkpoint['model']
            keys = list(checkpoint['model'].keys())
            for k in keys:
                if ('a_2' in k):
                    checkpoint['model'][k.replace('a_2', 'weight')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])
                if ('b_2' in k):
                    checkpoint['model'][k.replace('b_2', 'bias')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)



        self.to(device)

    def forward(self, src, tgt, query=None, para_order=None):
        tgt = tgt[:-1]
        batch_size, n_blocks, num_tokens = src.shape
        
        if self.args.model_type in ['query', 'heq', 'hero']:
            src_features, mask_hier = self.encoder(src, query)
        else:
            src_features, mask_hier = self.encoder(src)
        dec_state = self.decoder.init_decoder_state(src, src_features)
        if (self.args.hier):
            decoder_outputs = self.decoder(tgt, src_features.view(n_blocks, num_tokens, batch_size, -1).contiguous(), dec_state, memory_masks=mask_hier)
        else:
            decoder_outputs = self.decoder(tgt, src_features.view(n_blocks, num_tokens, batch_size, -1).contiguous(), dec_state)

        return decoder_outputs


