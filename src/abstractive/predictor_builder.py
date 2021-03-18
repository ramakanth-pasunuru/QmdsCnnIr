#!/usr/bin/env python
""" 
Translator Class and builder 
Majority of code borrowed from https://github.com/nlpyang/hiersumm
and https://github.com/OpenNMT/OpenNMT-py
"""
from __future__ import print_function
import codecs
import os
import math

import torch

from itertools import count

from tensorboardX import SummaryWriter

from abstractive.beam import GNMTGlobalScorer
from abstractive.cal_rouge import test_rouge, rouge_results_to_str
from abstractive.neural import tile
from abstractive.beam_search import BeamSearch

from rouge import FilesRouge

import numpy as np


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(alpha=args.alpha,beta=args.cov_beta,length_penalty='wu', coverage_penalty=args.coverage_penalty)
    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 n_best=1,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'
        self.args = args

        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']
        self.pad_token = symbols['PAD']


        self.block_ngram_repeat = self.args.block_ngram_repeat
        self.stepwise_penalty = self.args.stepwise_penalty

        self.n_best = n_best
        self.max_length = args.max_length
        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.dump_beam = dump_beam

        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = self.args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t<len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, attn, tgt_str, src = list(zip(*list(zip(translation_batch["predictions"],
                                                                         translation_batch["scores"],
                                                                         translation_batch["gold_score"],
                                                                         translation_batch["attention"],
                                                                         batch.tgt_str, batch.src))))


        translations = []
        for b in range(batch_size):
            pred_sents = sum([self._build_target_tokens(preds[b][n])
                for n in range(self.n_best)],[])
            gold_sent = tgt_str[b].split()
            if (self.args.hier):
                raw_src = '<PARA>'.join([self.vocab.DecodeIds(list([int(w) for w in t])) for t in src[b]])
            else:
                raw_src = self.vocab.DecodeIds(list([int(w) for w in src[b]]))

            translation = (pred_sents, gold_sent, raw_src, attn[b])
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations


    def translate(self,
                  data_iter,step):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold'%step
        can_path = self.args.result_path + '.%d.candidate'%step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        self.raw_gold_out_file = codecs.open(raw_gold_path, 'w', 'utf-8')
        self.raw_can_out_file = codecs.open(raw_can_path, 'w', 'utf-8')

        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        ct = 0
        with torch.no_grad():
            for batch in data_iter:

                with torch.no_grad():
                    batch_data = self._fast_translate_batch(
                        batch,
                        self.max_length,
                        min_length=self.min_length,
                        n_best=self.n_best)

                translations = self.from_batch(batch_data)

                for ind, trans in enumerate(translations):
                    pred, gold, src, attn = trans
                    pred_str = ' '.join(pred).replace('<Q>', ' ').replace(r' +', ' ').replace('<unk>', 'UNK').strip()
                    gold_str = ' '.join(gold).replace('<t>', '').replace('</t>', '').replace('<Q>', ' ').replace(r' +',
                                                                                                                 ' ').strip()
                    pred_str = ' '.join(pred_str.split()) # remove extra white spaces
                    gold_str = ' '.join(gold_str.split()) # remove extra white spaces


                    gold_str = gold_str.lower()
                    self.raw_can_out_file.write(' '.join(pred).strip() + '\n')
                    self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    ct += 1
                    if (ct > self.args.max_samples):
                        break

                self.raw_can_out_file.flush()
                self.raw_gold_out_file.flush()
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
                if (ct > self.args.max_samples):
                    break

        self.raw_can_out_file.close()
        self.raw_gold_out_file.close()
        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        if(step!=-1 and self.args.report_rouge):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s'%(step,rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)

            ## write the results to a file
            res_path = self.args.result_path + '.%d.result'%step
            res_out_file = codecs.open(res_path, 'w', 'utf-8')
            res_out_file.write(rouge_results_to_str(rouges))
            res_out_file.flush()
            res_out_file.close()

            return rouges


    def fast_rouge(self, step):
        self.logger.info("Calculating Rouge")

        gold_path = self.args.result_path + '.%d.gold'%step
        can_path = self.args.result_path + '.%d.candidate'%step

        if self.args.dataset in ["DUC2006", "DUC2007"]:
            ## give only one reference
            data = []
            with open(gold_path, 'r') as f:
                for line in f.read().splitlines():
                    data.append(line.strip())

            data = [d.split("<x>")[0].strip() for d in data]
            with open(gold_path, 'w') as f:
                f.write("\n".join(data))
                f.flush()

            print(8*"="+"DEBUG TEST FOR DUC"+8*"=")
            print(f"reference sample: {data[0]}")



        files_rouge = FilesRouge(can_path, gold_path)
        scores = files_rouge.get_scores(avg=True)

        rouges = {}
        rouges["rouge_l_f_score"] = scores["rouge-l"]["f"]
        rouges["rouge_2_f_score"] = scores["rouge-2"]["f"]
        rouges["rouge_1_f_score"] = scores["rouge-1"]["f"]
        self.logger.info(rouges)
 
        return rouges



    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        candidates = codecs.open(can_path, encoding="utf-8")
        references = codecs.open(gold_path, encoding="utf-8")
        if self.args.rouge_path is None:
            results_dict = test_rouge(candidates, references, 1)
        else:
            results_dict = test_rouge(candidates, references, 0, rouge_dir=os.path.join(os.getcwd(),self.args.rouge_path))
        return results_dict



    def translate_batch(self, batch,  fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length,
                n_best=self.n_best)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0,
                              n_best=1):
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size

        # Encoder forward.
        src = batch.src
        batch_size, n_blocks, num_tokens = src.shape
        
        if self.args.model_type in ['query', 'heq', 'hero']:
            src_features, mask_hier = self.model.encoder(src, batch.query)
        else:
            src_features, mask_hier = self.model.encoder(src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        src_features = src_features.view(n_blocks, num_tokens, batch_size, -1).contiguous()
        device = src_features.device


        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": [0] * batch_size}

         #(2) Repeat src objects `beam_size` times.
         # We use batch_size x beam_size
        
        dec_states.map_batch_fn(
                    lambda state, dim: tile(state, beam_size, dim=dim))


        src_features = tile(src_features, beam_size, dim=2)
        mask = tile(mask_hier, beam_size, dim=0)




        beam = BeamSearch(
            beam_size,
            n_best=self.n_best,
            batch_size=batch_size,
            global_scorer=self.global_scorer,
            pad=self.pad_token,
            eos=self.end_token,
            bos=self.start_token,
            min_length=self.min_length,
            ratio=0.,
            max_length=self.max_length,
            mb_device=device,
            return_attention=False,
            stepwise_penalty=self.stepwise_penalty,
            block_ngram_repeat=self.block_ngram_repeat,
            exclusion_tokens=set([]),
            memory_lengths=mask)

        for step in range(max_length):
            decoder_input = beam.current_predictions.view(1, -1)
            if (self.args.hier):
                dec_out, dec_states, attn = self.model.decoder(decoder_input, src_features, dec_states,
                                                         memory_masks=mask,
                                                         step=step)
            else:
                dec_out, dec_states, attn = self.model.decoder(decoder_input, src_features, dec_states,
                                                         step=step)

            # Generator forward.
            log_probs = self.generator.forward(dec_out.squeeze(0))
            vocab_size = log_probs.size(-1)
            attn = attn.transpose(1,2).transpose(0,1)
            attn = attn.max(2)[0]
            beam.advance(log_probs, attn)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin
            if any_beam_is_finished:
                src_features = src_features.index_select(2, select_indices)
                mask = mask.index_select(0, select_indices)

            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))
        
        results["scores"] = beam.scores
        results["predictions"] = beam.predictions
        results["attention"] = beam.attention
        
        return results


