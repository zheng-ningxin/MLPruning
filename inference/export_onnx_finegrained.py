import argparse
import glob
import json
import logging

import os
import random
import math
import copy
import numpy as np
import torch
# from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from nni.compression.pytorch.utils import get_module_by_name
from emmental import MaskedBertConfig, MaskedBertForSequenceClassification
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from emmental.modules.masked_nn import MaskedLinear

from nni.compression.pytorch import ModelSpeedup, apply_compression_results
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, LevelPruner
from nni.algorithms.compression.pytorch.pruning.weight_masker import WeightMasker
from nni.algorithms.compression.pytorch.pruning.dependency_aware_pruner import DependencyAwarePruner
from shape_hook import ShapeHook 

head_pruner_cfg = torch.load('head_prune_cfg')
task_name = "qqp"
model_name_or_path = '../training/result/qqp_partial/coarse_0.3/checkpoint-220000/'
data_dir = './QQP'
max_seq_length= 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_and_cache_examples( task, tokenizer, evaluate=False):
    
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        if task in ["mnli",
                    "mnli-mm"] and args.model_type in ["roberta",
                                                       "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(
                data_dir) if evaluate else processor.get_train_examples(
                data_dir))
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_seq_length,
            label_list=label_list,
            output_mode=output_mode,
        )
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels)
    return dataset


def evaluate(model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ['qqp']
    eval_outputs_dirs = './tmp'
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=32)

        # Eval!

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3]}
                
                inputs["token_type_ids"] = batch[2] 
                     # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                # import pdb; pdb.set_trace()
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        
        from scipy.special import softmax

        probs = softmax(preds, axis=-1)
        entropy = np.exp((-probs * np.log(probs)).sum(axis=-1).mean())
        preds = np.argmax(preds, axis=1)
      
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        if entropy is not None:
            result["eval_avg_entropy"] = entropy

        output_eval_file = os.path.join(
            eval_output_dir, prefix, "eval_results.txt")

    return results

from ModelVisual2 import ModelVisual
if __name__ == '__main__':
    prefix = 'bert_finegrained_0.95_onnx'
    os.makedirs('bert_finegrained_0.95_onnx', exist_ok=True)
    output_mode = output_modes[task_name]
    tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-QQP')
    config = BertConfig.from_pretrained('textattack/bert-base-uncased-QQP')
    model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-QQP', config=config)
    dummy_input = torch.load('dummy_input.pth')
    data = (dummy_input['input_ids'], dummy_input['attention_mask'], dummy_input['token_type_ids'])
    state_dict = torch.load('./bert_finegrained_0.9_ck/bert_finegrained_0.95_nomask.pth')
    model.load_state_dict(state_dict)
    fine_grained_mask = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w_mask = torch.abs(module.weight.data)>0
            b_mask = torch.abs(module.bias.data)>0
            fine_grained_mask[name] = {'weight':w_mask, 'bias':b_mask}
            print(name, ' Sparsity ratio:', torch.sum(w_mask)/w_mask.numel())
    
    import pdb; pdb.set_trace()
    torch.onnx.export(model, data, os.path.join(prefix, 'bert_finegrained.onnx'), opset_version=10)