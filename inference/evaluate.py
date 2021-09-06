import argparse
import glob
import json
import logging

import os
import random
import math

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

def load_and_cache_examples(args, task, tokenizer, evaluate=False):

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
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
                args.data_dir) if evaluate else processor.get_train_examples(
                args.data_dir))
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
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


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (
        "mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir +
                         "/MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size)

        # Eval!
        # print(f"***** Running evaluation {prefix} *****")
        # print(f"  Num examples = {len(eval_dataset)}")
        # print(f"  Batch size = {args.eval_batch_size}")
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in [
                            "bert", "masked_bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

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
        if args.output_mode == "classification":
            from scipy.special import softmax

            probs = softmax(preds, axis=-1)
            entropy = np.exp((-probs * np.log(probs)).sum(axis=-1).mean())
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            entropy = None
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        if entropy is not None:
            result["eval_avg_entropy"] = entropy

        output_eval_file = os.path.join(
            eval_output_dir, prefix, "eval_results.txt")

    return results

def load_weights_from_masked(bert, masked_bert):

    for name, module in masked_bert.named_modules():
        if isinstance(module, MaskedLinear):
            _, leaf_module = get_module_by_name(bert, name)
            assert isinstance(leaf_module, torch.nn.Linear)
            leaf_module.weight.data.copy_(module.weight.data)
            mask_head, mask = module.get_mask()

            weight_shape = module.weight.size()
            bias_shape = module.bias.size()
            if mask_head is not None:
                weight_thresholded = (
                    module.weight.view(
                        module.head_split, -1) * mask_head).view(weight_shape)
                if module.bias_mask:
                    bias_thresholded = (
                        module.bias.view(
                            module.head_split, -1) * mask_head).view(bias_shape)
            else:
                weight_thresholded = module.weight.data
                bias_thresholded = module.bias.data
            # Mask weights with computed mask
            if mask is not None:
                weight_thresholded = mask * weight_thresholded
                if module.bias_mask:
                    bias_thresholded = mask.view(
                        module.bias.size()) * bias_thresholded
                else:
                    bias_thresholded = bias_thresholded
            pass
            with torch.no_grad():
                leaf_module.weight.data.copy_(weight_thresholded.data)
                if leaf_module.bias is not None:
                    leaf_module.bias.data.copy_(bias_thresholded)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name_or_path')
    # parser.add_argument('--block_path')
    parser.add_argument(
        "--task_name",
        default="qqp",
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(
             processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument('--data_dir', default='../../data-bin/glue_data/QQP')
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--pruning_method",
        default="topK",
        type=str,
        help="Pruning Method (l0 = L0 regularization, magnitude = Magnitude pruning, topK = Movement pruning, sigmoied_threshold = Soft movement pruning).",
    )
    parser.add_argument(
        "--head_pruning", action="store_true", help="Head Pruning or not",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-
                        1, help="For distributed training: local_rank")

    parser.add_argument(
        "--block_rows",
        type=int,
        default=-1,
        help="Number of rows in a block")
    parser.add_argument(
        "--block_cols",
        type=int,
        default=-1,
        help="Number of cols in a block")
    parser.add_argument(
        "--block_path",
        default=None,
        type=str,
        help="Path to pretrained block wise model",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True
    )
    args = parser.parse_args()
    args.device = torch.device('cuda')
    args.output_mode = output_modes[args.task_name]
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    config = MaskedBertConfig.from_pretrained(args.model_name_or_path)
    model = MaskedBertForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config).to(args.device)
    #result = evaluate(args, model, tokenizer)
    #print(result)

    norm_model = BertForSequenceClassification(config=config)
    load_weights_from_masked(norm_model, model)
    result = evaluate(args, model, tokenizer)
    print(result)
    import pdb; pdb.set_trace()
    pass
    # prune heads for the norm model
    head_pruner_cfg ={}
    n_layers = len(norm_model.bert.encoder.layer)
    for layer_id in range(n_layers):
        head_pruner_cfg[layer_id] = []
        q_layer = norm_model.bert.encoder.layer[layer_id].attention.self.query
        head_step = q_layer.weight.size(0) // 12
        for i in range(12):
            _start = i * head_step
            _end = _start + head_step
            if torch.sum(q_layer.weight.data[_start:_end]) < 1e-8:
                head_pruner_cfg[layer_id].append(i)
    norm_model.prune_heads(head_pruner_cfg)
    norm_model = norm_model.to(args.device)
    import pdb; pdb.set_trace()
    print('Bert Head Pruning config', head_pruner_cfg)
    new_result = evaluate(args, norm_model, tokenizer)
    print(new_result)
                
    

if __name__ == '__main__':
    main()
