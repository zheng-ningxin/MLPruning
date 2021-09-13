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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)




def regularization(model: nn.Module, threshold: float):
    threshold_list = []
    for name, param in model.named_parameters():
        if 'threshold' in name:
            threshold_list.append(torch.sigmoid(param))
    # BERT-base has 12 layers
    param_remain = 0
    layer_num = 12
    block_num = len(threshold_list) // layer_num
    for i in range(12):
        param_remain += remain_param_compute(
            threshold_list[i * block_num: (i + 1) * block_num])

    if param_remain / 144. - threshold <= 0:
        reg_loss = param_remain * 0.
    else:
        # 144 comes from count, use simple sqaure loss
        reg_loss = torch.square(param_remain / 144. - threshold)

    return reg_loss

def train(args, train_dataset, model, tokenizer, teacher=None):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size)

    t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "mask_score" in n or "threshold" in n and p.requires_grad],
            "lr": args.mask_scores_learning_rate,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n and "threshold" not in n and p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n and "threshold" not in n and p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "lr": args.learning_rate,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)



    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed) = {args.train_batch_size}",

    )
    print(f"  Total optimization steps = {t_total}")
    # Distillation
    if teacher is not None:
        print("  Training with distillation")

    global_step = 0

    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproducibility

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "masked_bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(**inputs)
                # print(outputs)

                if "masked" not in args.model_type:
                    # model outputs are always tuple in transformers (see doc)
                    loss, logits_stu = outputs.loss, outputs.logits
                else:
                    loss, logits_stu, reps_stu, attentions_stu = outputs

                # Distillation loss
                if teacher is not None:
                    if "token_type_ids" not in inputs:
                        inputs["token_type_ids"] = None if args.teacher_type == "xlm" else batch[2]
                    with torch.no_grad():
                        outputs_tea = teacher(
                            input_ids=inputs["input_ids"],
                            token_type_ids=inputs["token_type_ids"],
                            attention_mask=inputs["attention_mask"],
                        )
                        logits_tea = outputs_tea.logits
                        reps_tea, attentions_tea = outputs_tea.hidden_states, outputs_tea.attentions

                    teacher_layer_num = len(attentions_tea)
                    student_layer_num = len(attentions_stu)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(
                        teacher_layer_num / student_layer_num)
                    new_attentions_tea = [attentions_tea[i *
                                                         layers_per_block +
                                                         layers_per_block -
                                                         1] for i in range(student_layer_num)]

                    att_loss, rep_loss = 0, 0
                    for student_att, teacher_att in zip(
                            attentions_stu, new_attentions_tea):
                        student_att = torch.where(
                            student_att <= -1e2,
                            torch.zeros_like(student_att).to(
                                args.device),
                            student_att)
                        teacher_att = torch.where(
                            teacher_att <= -1e2,
                            torch.zeros_like(teacher_att).to(
                                args.device),
                            teacher_att)

                        tmp_loss = F.mse_loss(
                            student_att, teacher_att, reduction="mean",)
                        att_loss += tmp_loss

                    new_reps_tea = [reps_tea[i * layers_per_block]
                                    for i in range(student_layer_num + 1)]
                    new_reps_stu = reps_stu
                    for i_threp, (student_rep, teacher_rep) in enumerate(
                            zip(new_reps_stu, new_reps_tea)):
                        tmp_loss = F.mse_loss(
                            student_rep, teacher_rep, reduction="mean",)
                        rep_loss += tmp_loss

                    loss_logits = F.kl_div(
                        input=F.log_softmax(logits_stu / args.temperature, dim=-1),
                        target=F.softmax(logits_tea / args.temperature, dim=-1),
                        reduction="batchmean",
                    ) * (args.temperature ** 2)

                    loss_distill = loss_logits + rep_loss + att_loss
                    loss = args.alpha_distil * loss_distill + args.alpha_ce * loss

            if args.final_threshold < 1:
                # for pruning
                regu_ = regularization(
                    model=model, threshold=args.final_threshold)
                regu_lambda = max(args.final_lambda * regu_.item() /
                                  (1 - args.final_threshold) / (1 - args.final_threshold), 50)
                if regu_.item() < 0.0003:
                    # when the loss is very small, no need to pubnish it too
                    # much
                    regu_lambda = 10.
            else:
                # For baseline training
                regu_ = 0
                regu_lambda = 0

            loss = loss + regu_lambda * regu_

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if True:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [
                    -1,
                        0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    # Only evaluate when single GPU otherwise metrics may not
                    # average well
                    if (args.local_rank == -
                            1 and args.evaluate_during_training):
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()
                    logs["learning_rate"] = learning_rate_scalar[0]
                    if len(learning_rate_scalar) > 1:
                        for idx, lr in enumerate(learning_rate_scalar[1:]):
                            logs[f"learning_rate/{idx+1}"] = lr
                    logs["loss"] = loss_scalar
                    if teacher is not None:
                        logs["loss/distil"] = loss_distill.item()
                        logs["loss/distil_logits"] = loss_logits.item()
                        try:
                            logs["loss/distil_attns"] = att_loss.item()
                        except BaseException:
                            logs["loss/distil_attns"] = 0
                        try:
                            logs["loss/distil_reps"] = rep_loss.item()
                        except BaseException:
                            logs["loss/distil_reps"] = 0

                    logging_loss = tr_loss
                    print(f"step: {global_step}: {logs}")

                if args.local_rank in [-1,
                                       0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(
                        args, os.path.join(
                            output_dir, "training_args.bin"))
                    print(f"Saving model checkpoint to {output_dir}")

                    torch.save(
                        optimizer.state_dict(), os.path.join(
                            output_dir, "optimizer.pt"))
                    torch.save(
                        scheduler.state_dict(), os.path.join(
                            output_dir, "scheduler.pt"))
                    if args.fp16:
                        torch.save(
                            scaler.state_dict(), os.path.join(
                                output_dir, "scaler.pt"))

                    print(
                        f"Saving optimizer and scheduler states to {output_dir}")

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


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

def debug(model1, model2, dummy_input):
    def forward_hook(module, inputs, output):
        module.debug_out = output
        module.debug_in = inputs
    model1.eval()
    model2.eval()
    model1_hook = []
    model2_hook = []
    for name, module in model1.named_modules():
        hook = module.register_forward_hook(forward_hook)
        model1_hook.append(hook)
    for name, module in model2.named_modules():
        hook = module.register_forward_hook(forward_hook)
        model2_hook.append(hook)
    with torch.no_grad():

            out1 = model1(**dummy_input)
            out2 = model2(**dummy_input)
    outs_1 = {}
    outs_2 = {}
    name2_module1 = {n:m for n,m in model1.named_modules()}
    for name, module in model1.named_modules():
        if hasattr(module, 'debug_out') and isinstance(module.debug_out, torch.Tensor):
            _sum = torch.sum(module.debug_out).item()
            outs_1[name] = _sum         
    for name, module in model2.named_modules():
        if hasattr(module, 'debug_out') and  isinstance(module.debug_out, torch.Tensor):
            _sum = torch.sum(module.debug_out).item()
            outs_2[name] = _sum
    for name in outs_1:
        if outs_1[name] != outs_2[name]:
            print(name, 'Type', type(name2_module1[name]))
            print(outs_1[name], outs_2[name])

def get_dummy_input(args, model, tokenizer):
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
                return inputs

def export_model(model):
    mask = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            mask[name] = {'weight_mask':torch.ones_like(module.weight), 'bias_mask':torch.ones_like(module.bias)}
            mask[name]['weight_mask'][module.weight==0] = 0
            mask[name]['bias_mask'][module.bias==0] = 0
    torch.save(mask, './mask')
    torch.save(model.state_dict(), './weight')

    
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
    parser.add_argument('--data_dir', default='./QQP')
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
    
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")

    # Pruning parameters
    parser.add_argument(
        "--mask_scores_learning_rate",
        default=1e-2,
        type=float,
        help="The Adam initial learning rate of the mask scores.",
    )
    parser.add_argument(
        "--final_threshold",
        default=0.7,
        type=float,
        help="Final value of the threshold (for scheduling).")


    parser.add_argument(
        "--mask_init",
        default="constant",
        type=str,
        help="Initialization method for the mask scores. Choices: constant, uniform, kaiming.",
    )
    parser.add_argument(
        "--mask_scale",
        default=0.0,
        type=float,
        help="Initialization parameter for the chosen initialization method.")

    parser.add_argument(
        "--final_lambda",
        default=0.0,
        type=float,
        help="Regularization intensity (used in conjunction with `regularization`.",
    )

    # Distillation parameters (optional)
    parser.add_argument(
        "--teacher_type",
        default=None,
        type=str,
        help="Teacher type. Teacher tokenizer and student (model) tokenizer must output the same tokenization. Only for distillation.",
    )
    parser.add_argument(
        "--teacher_name_or_path",
        default=None,
        type=str,
        help="Path to the already fine-tuned teacher model. Only for distillation.",
    )
    parser.add_argument(
        "--alpha_ce",
        default=0.1,
        type=float,
        help="Cross entropy loss linear weight. Only for distillation.")
    parser.add_argument(
        "--alpha_distil",
        default=0.9,
        type=float,
        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument(
        "--temperature",
        default=2.0,
        type=float,
        help="Distillation temperature. Only for distillation.")

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
    "--warmup_steps",
    default=0,
    type=int,
    help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
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
    evaluate(args, norm_model, tokenizer)
    torch.save('pretrained_bert_qqp.bert')

if __name__ == '__main__':
    main()
