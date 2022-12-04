import torch
import random
from collections import OrderedDict
import json
from tabulate import tabulate
from typing import Dict, List, Tuple, Optional
import numpy as np
import re

from constants import *


def format_typed_mention(template):
    t = template["type"]
    mentions = []
    if t == "objective" or t == "objvar":
        if "direction" in template and template["direction"] is not None:
            mentions.append([START_OF_OBJ_DIR, " " + template["direction"].strip(" ") +" ", END_OF_OBJ_DIR])
        if "name" in template and template["name"] is not None:
            mentions.append([START_OF_OBJ_NAME, " " + template["name"].strip(" ") + " ", END_OF_OBJ_NAME])
        if "terms" in template:
            mentions.append([" " + IS_TOKEN + " "])
            # TODO: Create a template chunk for each arrangement of variables in terms
            for var, param in template["terms"].items():
                mentions.append([
                    START_OF_VAR, " " + var.strip(" ") + " ", END_OF_VAR,
                    " [TIMES] ",
                    START_OF_PARAM, " " + param.strip(" ") + " ", END_OF_PARAM  # TODO: verify
                ])
        if "vars" in template:
            mentions.append([" " + IS_TOKEN + " "])
            # TODO: Create a template chunk for each arrangement of variables in terms
            for var in template["vars"]:
                mentions.append([
                    START_OF_VAR, " " + var.strip(" ") + " ", END_OF_VAR,
                    " [TIMES] ",
                    START_OF_PARAM, " " + ONE_TOKEN + " ", END_OF_PARAM
                ])


    else:
        if "direction" in template:
            mentions.append([START_OF_CONST_DIR, " " + template["direction"].strip(" ") + " ", END_OF_CONST_DIR])
        
        if "operator" in template:
            mentions.append([START_OF_OPERATOR, " " + template["operator"].strip(" ") + " ", END_OF_OPERATOR])

        if "limit" in template:
            mentions.append([START_OF_LIMIT, " " + template["limit"].strip(" ") + " ", END_OF_LIMIT])

        if t == "upperbound":
            mentions.append([START_OF_CONST_TYPE, " " + TYPE_UPPER_BOUND + " ", END_OF_CONST_TYPE])
            if "var" in template:
                mentions.append([" " + FOR_TOKEN + " "])
                mentions.append([START_OF_VAR, " " + template["var"].strip(" ") + " ", END_OF_VAR])

        elif t == "lowerbound":
            mentions.append([START_OF_CONST_TYPE, " " + TYPE_LOWER_BOUND + " ", END_OF_CONST_TYPE])
            if "var" in template:
                mentions.append([" " + FOR_TOKEN + " "])
                mentions.append([START_OF_VAR, " " + template["var"].strip(" ") + " ", END_OF_VAR])

        elif t == "sum":
            mentions.append([START_OF_CONST_TYPE, " " + TYPE_SUM_CONST + " ", END_OF_CONST_TYPE])

        elif t == "linear":
            mentions.append([START_OF_CONST_TYPE, " " + TYPE_LINEAR_CONST + " ", END_OF_CONST_TYPE])
            mentions.append([" " + IS_TOKEN + " "])
            # TODO: Create a template chunk for each arrangement of variables in terms
            for var, param in template["terms"].items():
                mentions.append([
                    START_OF_VAR, " " + var.strip(" ") + " ", END_OF_VAR,
                    " [TIMES] ",
                    START_OF_PARAM, " " + param.strip(" ") + " ", END_OF_PARAM  # TODO: verify
                ])

        elif t == "ratio":
            mentions.append([START_OF_CONST_TYPE," " + TYPE_RATIO_CONST + " ", END_OF_CONST_TYPE])
            if "var" in template:
                mentions.append([" " + FOR_TOKEN + " "])
                mentions.append([START_OF_VAR, " " + template["var"].strip(" ") + " ", END_OF_VAR])
        elif t == "xby":
            mentions.append([START_OF_CONST_TYPE, " " + TYPE_XBY_CONST + " ", END_OF_CONST_TYPE])
            if "y_var" in template and "param" in template:
                mentions.append([
                    START_OF_VAR, " " + template["y_var"].strip(" ") + " ", END_OF_VAR,
                    " [TIMES] ",
                    START_OF_PARAM, " " + template["param"].strip(" ") + " ", END_OF_PARAM  # TODO: verify
                ])
            if "x_var" in template:
                mentions.append([" " + IS_TOKEN + " "])
                mentions.append([START_OF_VAR, " " + template["x_var"].strip(" ") + " ", END_OF_VAR])
        elif t == "xy":
            mentions.append([START_OF_CONST_TYPE, " " + TYPE_XY_CONST + " ", END_OF_CONST_TYPE])
            if "y_var" in template:
                mentions.append([
                    START_OF_VAR, " " + template["y_var"].strip(" ") + " ", END_OF_VAR
                ])
            if "x_var" in template:
                mentions.append([" " + IS_TOKEN + " "])
                mentions.append([START_OF_VAR, " " + template["x_var"].strip(" ") + " ", END_OF_VAR])
        else:
            raise ValueError("Not implemented")

    return mentions



def token2sub_tokens(tokenizer, token):
    """
    Take in a string value and use tokenizer to tokenize it into subtokens.
    Return a list of sub tokens.
    """
    res = []
    for sub_token in tokenizer.tokenize(token):
        # make sure it's not an empty string
        if len(sub_token) > 0:
            res.append(tokenizer.convert_tokens_to_ids(sub_token))
    return res


def format_inputs_outputs(flattened_seqs, tokenizer, use_gpu, max_position_embeddings, replace_pad_tokens):
    max_seq_len = max([len(seq) for seq in flattened_seqs])
    max_seq_len = min(max_position_embeddings, max_seq_len)

    # create padding & mask
    decoder_input_ids = []
    decoder_masks = []
    decoder_labels = []

    for flattened_seq in flattened_seqs:
        mask = [1] * len(flattened_seq) + [0] * (max_seq_len - len(flattened_seq) - 1)

        # padding.
        flattened_seq += [tokenizer.pad_token_id] * (max_seq_len - len(flattened_seq))

        mask = mask[:max_seq_len - 1]
        flattened_seq = flattened_seq[:max_seq_len]

        input_ids = flattened_seq[:-1]
        labels = flattened_seq[1:]

        if replace_pad_tokens:
            labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

        decoder_input_ids.append(input_ids)
        decoder_labels.append(labels)
        decoder_masks.append(mask)

    # form tensor
    if use_gpu:
        decoder_input_ids = torch.cuda.LongTensor(decoder_input_ids)
        decoder_labels = torch.cuda.LongTensor(decoder_labels)
        decoder_masks = torch.cuda.FloatTensor(decoder_masks)

    else:
        decoder_input_ids = torch.LongTensor(decoder_input_ids)
        decoder_labels = torch.LongTensor(decoder_labels)
        decoder_masks = torch.FloatTensor(decoder_masks)

    res = {
        'decoder_input_ids': decoder_input_ids,
        'decoder_labels': decoder_labels,
        'decoder_masks': decoder_masks
    }
    return res


def generate_decoder_inputs_outputs(batch, tokenizer, model, use_gpu, max_position_embeddings, replace_pad_tokens=True, natural_parsing=False):
    '''
    Process decoder_input_chunks and produce a dictionary with keys decoder_input_ids and decoder_labels.
    decoder_input_chunks is a list where each element correspond to annotation of a document.
    '''
    if not model.bert.config.name_or_path.startswith('facebook/bart'):
        print("model name ", model.bert.config)
        raise NotImplementedError

    decoder_input_chunks = batch.decoder_input_chunks

    # TODO: Do we generate entire LP formulation or only one declaration at a time?
    flattened_seqs = []

    for decoder_input_chunk in decoder_input_chunks: # decoder_input_chunks: 4D, decoder_input_chunk: 3D
        flatten_entities = []

        for template in decoder_input_chunk: # decoder_input_chunk: 3D, template: 2D
            flatten_entities.append(tokenizer.convert_tokens_to_ids(START_OF_TEMPLATE))
            for entity in template: # template 2D, 
                if natural_parsing:
                    flatten_entities.append(entity)
                else:
                    for sub_token in entity:
                        flatten_entities.append(sub_token)
            flatten_entities.append(tokenizer.convert_tokens_to_ids(END_OF_TEMPLATE))
        flattened_seq = [model.bert.config.decoder_start_token_id, tokenizer.bos_token_id] + flatten_entities + [
            tokenizer.eos_token_id]

        flattened_seqs.append(flattened_seq)

    res = format_inputs_outputs(flattened_seqs, tokenizer, use_gpu, max_position_embeddings, replace_pad_tokens)

    return res


# From: https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/utilities/seed.py
def pl_worker_init_function(worker_id: int, rank: Optional[int] = None) -> None:  # pragma: no cover
    """The worker_init_fn that Lightning automatically adds to your dataloader if you previously set the seed with
    ``seed_everything(seed, workers=True)``.
    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    global_rank = rank if rank is not None else rank_zero_only.rank
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    log.debug(
        f"Initializing random number generators of process {global_rank} worker {worker_id} with base seed {base_seed}"
    )
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)


def canonicalize_objective(obj):
    if "terms" in obj:
        termstr = []
        for var, multiplier in obj["terms"].items():
            multiplier = multiplier.rstrip(".")
            termstr.append(f"{multiplier} times {var}")

        if len(termstr) > 0:
            termstr = ", ".join(termstr)
        else:
            termstr = EMPTY_TOKEN
        objdef = f"the sum of {termstr}"
    elif "vars" in obj:
        termstr = []
        for var in obj["vars"]:
            termstr.append(f"1 times {var}")

        if len(termstr) > 0:
            termstr = ", ".join(termstr)
        else:
            termstr = EMPTY_TOKEN
        objdef = f"the sum of {termstr}"
    else:
        raise Exception(f"{obj} contains neither terms nor vars")

    canobj = "The objective is %s which is defined as %s. %s the objective." % (
        obj["name"],
        objdef,
        obj["direction"],
    )
    
    return canobj


def canonicalize_constraint(const):
    output = "Constraint "
    if "type" in const:
        output += f"of type {const['type']} "
    else:
        raise Exception(f"type is not defined in {const}")
        
    if "direction" in const:
        output += f"with direction {const['direction']} "
        
    output += "is that "
    
    if const["type"] == "linear":
        if "terms" in const:
            termsarr = []
            for var, multiplier in const["terms"].items():
                termsarr.append(f"{multiplier} times {var}")

            if len(termsarr) > 0:
                termsstr = ", ".join(termsarr)
            else:
                termsstr = EMPTY_TOKEN
            termsstr = "sum of " + termsstr + " is " + const["operator"] + " to " + const["limit"]
            output += termsstr
        else:
            raise Exception(f"terms is not defined in {const}")
    elif const["type"] == "sum":
        output += f"the sum is {const['operator']} to {const['limit']}"
    elif const["type"] == "lowerbound":
        output += f"{const['var']} is {const['operator']} to {const['limit']}"
    elif const["type"] == "xby":
        output += f"{const['x_var']} is {const['operator']} to {const['param']} times of {const['y_var']}"
    elif const["type"] == "ratio":
        output += f"{const['var']} is {const['operator']} to {const['limit']} of total"
    elif const["type"] == "upperbound":
        output += f"{const['var']} is {const['operator']} to {const['limit']}"
    elif const["type"] == "xy":
        output += f"{const['x_var']} is {const['operator']} to {const['y_var']}"
    else:
        raise Exception(f"{const['type']} is not defined.")

    return output


def canonicalize_template(template):
    t = template["type"]
    if t == "objective" or t == "objvar":
        return canonicalize_objective(template)
    else:
        return canonicalize_constraint(template)
