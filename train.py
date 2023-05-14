import os
import sys
import json
import time
import random
from argparse import ArgumentParser

import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, Adafactor, set_seed
from model import TextMappingModel
from config import Config
from data import LPMappingDataset
from data_per_declaration import DeclarationMappingDataset
from constants import *
from utils import *
import test_utils

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='configs/default.json')
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()
config = Config.from_json_file(args.config)

# fix random seed
if args.seed is not None:
    config.seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)

print(f"Config: {config.to_dict()}")

# increase recursion limit
max_length = int(config.max_length)
sys.setrecursionlimit(max_length  * max_length + 100)

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# output
timestamp = os.getenv('NL4OPT_TIMESTAMP')
if timestamp is None:
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

log_dir = os.path.join(config.log_path, timestamp)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# logger = Logger(log_dir)
output_dir = os.path.join(config.output_path, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_file = os.path.join(output_dir, 'log.txt')
with open(log_file, 'w', encoding='utf-8') as w:
    w.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))
best_model = os.path.join(output_dir, 'best.mdl')
train_result_file = os.path.join(output_dir, 'result.train.json')
dev_result_file = os.path.join(output_dir, 'result.dev.json')
test_result_file = os.path.join(output_dir, 'result.test.json')

# datasets
model_name = config.bert_model_name

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir)

tokenizer.add_tokens(SPECIAL_TOKENS)

if config.per_declaration:
    print('==============Prepare Training Set=================')
    train_set = DeclarationMappingDataset(config.train_file, max_length=config.max_length, gpu=use_gpu,
                                          enrich_ner=config.enrich_ner)
    print('==============Prepare Dev Set=================')
    dev_set = DeclarationMappingDataset(config.dev_file, max_length=config.max_length, gpu=use_gpu,
                                        enrich_ner=config.enrich_ner)
else:
    print('==============Prepare Training Set=================')
    train_set = LPMappingDataset(config.train_file, tokenizer=tokenizer, max_length=config.max_length, gpu=use_gpu, enrich_ner=config.enrich_ner)
    print('==============Prepare Dev Set=================')
    dev_set = LPMappingDataset(config.dev_file, tokenizer=tokenizer, max_length=config.max_length, gpu=use_gpu, enrich_ner=config.enrich_ner)

if config.per_declaration:
    print('==============Prepare Training Set=================')
    train_set.numberize(tokenizer)
    print('==============Prepare Dev Set=================')
    dev_set.numberize(tokenizer)
else:
    print('==============Prepare Training Set=================')
    train_set.numberize()
    print('==============Prepare Dev Set=================')
    dev_set.numberize()

# TODO: define dev and test golds

batch_num = len(train_set) // (config.batch_size * config.accumulate_step) + \
            (len(train_set) % (config.batch_size * config.accumulate_step) != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + \
                (len(dev_set) % config.eval_batch_size != 0)
# test_batch_num = len(test_set) // config.eval_batch_size + \
#                  (len(test_set) % config.eval_batch_size != 0)

# initialize the model

model = TextMappingModel(config)

model.load_bert(model_name, cache_dir=config.bert_cache_dir, tokenizer=tokenizer)

if not model_name.startswith('roberta'):
    print(f"Resizing embedding matrix.")
    model.bert.resize_token_embeddings(len(tokenizer))

if use_gpu:
    model.cuda(device=config.gpu_device)

# optimizer
optimizer = AdamW(params=model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * config.warmup_epoch,
                                           num_training_steps=batch_num * config.max_epoch)

# model state
state = dict(model=model.state_dict(), config=config.to_dict())

best_dev = -np.inf
current_step = 0
best_epoch = 0
best_score = 0

print('================Start Training================')
for epoch in range(config.max_epoch):

    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='Train {}'.format(epoch))
    optimizer.zero_grad()
    train_gold_outputs, train_pred_outputs, train_input_tokens, train_doc_ids, train_input_ids = [], [], [], [], []
    training_loss = 0
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size,
            shuffle=True, drop_last=False, collate_fn=train_set.collate_fn, num_workers=0)):

        decoder_inputs_outputs = generate_decoder_inputs_outputs(batch, tokenizer, model, use_gpu,
                                                                 config.max_position_embeddings,
                                                                 replace_pad_tokens=config.use_copy)
        decoder_input_ids = decoder_inputs_outputs['decoder_input_ids']

        decoder_labels = decoder_inputs_outputs['decoder_labels']
        decoder_masks = decoder_inputs_outputs['decoder_masks']

        loss = model(batch, decoder_input_ids, decoder_labels, tokenizer=tokenizer)['loss']
        # print(f"train_loss: {loss}")
        current_step += 1
        loss = loss * (1 / config.accumulate_step)
        training_loss += loss.item()
        loss.backward()

        train_gold_outputs.extend(decoder_inputs_outputs['decoder_labels'].tolist())
        train_input_ids.extend(decoder_input_ids.tolist())

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    # train the last batch
    if batch_num % config.accumulate_step != 0:
        progress.update(1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()


    train_result = {
        'pred_outputs': train_pred_outputs,
        'gold_outputs': train_gold_outputs,
        'pred_texts': [tokenizer.decode(train_pred_outputs[i]) for i in range(len(train_pred_outputs))],
        'gold_texts': [tokenizer.decode(train_gold_outputs[i]) for i in range(len(train_gold_outputs))],
        'input_tokens': train_input_tokens,
        'decoder_input_ids': train_input_ids,
        'doc_ids': train_doc_ids
    }
    with open(train_result_file + f'_{epoch}', 'w') as f:
        f.write(json.dumps(train_result))

    progress.close()

    print("training loss", training_loss)

    if epoch % config.eval_period == 0 or epoch == config.max_epoch - 1:
        # set print_errors to false as model is not expected to make correct syntax on early epochs
        dev_result = test_utils.evaluate(
            tokenizer,
            model,
            dev_set,
            epoch,
            dev_batch_num,
            use_gpu,
            config,
            tqdm_descr='Dev {}'.format(epoch),
            print_errors=False,
            per_declaration=config.per_declaration,
        )

        print(f"dev_accuracy: {dev_result['accuracy']}")

        with open(dev_result_file + f'_{epoch}', 'w') as f:
            f.write(json.dumps(dev_result))

        # save best result
        if dev_result['accuracy'] >= best_score:
            best_epoch = epoch
            best_score = dev_result['accuracy']
            print("Saving model with best dev set accuracy:", dev_result['accuracy'])
            state = dict(model=model.state_dict(), config=config.to_dict())
            model_output_path = os.path.join(output_dir, 'best-checkpoint.mdl')
            torch.save(state, model_output_path)

        # Save last checkpoint
        state = dict(model=model.state_dict(), config=config.to_dict())
        model_output_path = os.path.join(output_dir, 'checkpoint-last.mdl')
        torch.save(state, model_output_path)

        print('Log file', log_file)

print(f'Best epoch: {best_epoch} with accuracy {best_score}. Corresponding model saved as best-checkpoint.mdl')
print(config.to_dict())
