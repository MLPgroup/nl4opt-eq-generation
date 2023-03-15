import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers import BeamSearchScorer, LogitsProcessorList, NoBadWordsLogitsProcessor, ForcedBOSTokenLogitsProcessor, MaxLengthCriteria
from generation_bart import CopyConditionalGeneration
from constants import *

""" Model for text-to-text mapping using BART with copy mechanism for conditional generation

"""
class TextMappingModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # BERT encoder
        self.max_position_embeddings = config.max_position_embeddings
        self.SOT_weights = config.SOT_weights
        self.use_copy = config.use_copy
        self._k = config.k

    def load_bert(self, name, cache_dir=None, tokenizer=None):
        """Load the pre-trained LM (used in training phrase)
        :param name (str): pre-trained LM name
        :param cache_dir (str): path to the LM cache directory
        """
        print('Loading pre-trained LM {}'.format(name))

        if self.use_copy:
            self.bert = CopyConditionalGeneration.from_pretrained(name, cache_dir=cache_dir,
                                                                         output_attentions=True)
            self.bert._k = self._k
        else:
            self.bert = AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=cache_dir)

    def forward(self, batch, decoder_input_ids=None, decoder_labels=None, decoder_masks=None, logger=None, tag=None,
                step=None, tokenizer=None):

        res = {}
        vocab_size = len(tokenizer)

        weight = torch.ones(vocab_size).to(batch.input_ids.device)
        self.bert._loss_weight = weight
        self.bert._vocab_size = vocab_size

        if self.use_copy:
            bart_outputs = self.encode(batch, decoder_input_ids=decoder_input_ids, decoder_labels=decoder_labels)
        else:
            bart_outputs = self.encode(batch, decoder_input_ids=decoder_input_ids)

        if decoder_labels is not None:

            if self.use_copy:
                loss = bart_outputs.loss
            else:
                weight[tokenizer.convert_tokens_to_ids(START_OF_TEMPLATE)] = self.SOT_weights
                loss = torch.nn.functional.cross_entropy(input=bart_outputs.logits.view(-1, vocab_size),
                                                         target=decoder_labels.view(-1), weight=weight)

            res['loss'] = loss

        return res

    def encode(self, batch, decoder_input_ids=None, decoder_labels=None, decoder_masks=None):
        '''
        Encode the input documents
        '''

        return self.bert(input_ids=batch.input_ids,
                         attention_mask=batch.attention_masks,
                         # 1 for tokens that are not masked, 0 for tokens that are masked.
                         decoder_input_ids=decoder_input_ids,
                         # For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided, the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
                         labels=decoder_labels,
                         # decoder_attention_mask=decoder_masks, #Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
                         return_dict=True,
                         output_hidden_states=True,

                         )

    def generate(self, batch, num_beams, decoding_length):
        '''
        From https://huggingface.co/transformers/main_classes/model.html?highlight=beamsearchscorer
        '''

        self.bert._cache_input_ids = batch.input_ids.repeat_interleave(num_beams, dim = 0)
        return self.bert.generate(
            input_ids = batch.input_ids,
            attention_mask = batch.attention_masks,
            max_length = decoding_length,
            early_stopping = False,
            num_beams = num_beams,
            no_repeat_ngram_size = 0,
        )


    def predict(self, batch, tokenizer, epoch=None, beam_size=1):
        self.eval()

        with torch.no_grad():

            decoding_length = self.max_position_embeddings - 1
            decoded_ids = self.generate(batch, num_beams=beam_size, decoding_length=decoding_length)

            res = {
                'decoded_ids': decoded_ids
            }

        self.train()
        return res

