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

from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer

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
import nni
import torch
import sys
import os
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from nni_finetune import *

def get_optimizer_scheduler(model, train_dataloader):
    num_train_epochs = 20
    t_total = len(train_dataloader) * num_train_epochs
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    weight_decay = 0.0
    warmup_steps=100
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n and "threshold" not in n and p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n and "threshold" not in n and p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "lr": learning_rate,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total)
    return optimizer, scheduler


def train(train_dataset, model, tokenizer, optimizer, scheduler, teacher=None, num_train_epochs=10):
    """ Train the model """
    train_sampler = RandomSampler(
        train_dataset) 
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=32)

    t_total = len(train_dataloader) * num_train_epochs
    temperature = 2.0
    alpha_distil = 0.9
    alpha_ce =0.1
    max_grad_norm = 1.0
    # Prepare optimizer and schedule (linear warmup and decay)

    # Train!
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")

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
    # train_iterator = trange(
    #     epochs_trained,
    #     int(num_train_epochs),
    #     desc="Epoch",

    # )
    # Added here for reproducibility
    best_acc = 0

    for _ in range(num_train_epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]}
            
            inputs["token_type_ids"] = batch[2] # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(**inputs)
                # print(outputs)

                loss, logits_stu, reps_stu, attentions_stu = outputs[0], outputs[1], outputs[2], outputs[3]
                # import pdb; pdb.set_trace()
                # Distillation loss
                if teacher is not None:
                    if "token_type_ids" not in inputs:
                        inputs["token_type_ids"] = batch[2]
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
                            torch.zeros_like(student_att).to(device),
                            student_att)
                        teacher_att = torch.where(
                            teacher_att <= -1e2,
                            torch.zeros_like(teacher_att).to(device),
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
                        input=F.log_softmax(logits_stu / temperature, dim=-1),
                        target=F.softmax(logits_tea / temperature, dim=-1),
                        reduction="batchmean",
                    ) * (temperature ** 2)

                    loss_distill = loss_logits + rep_loss + att_loss
                    loss = alpha_distil * loss_distill + alpha_ce * loss




         
            loss.backward()

            tr_loss += loss.item()
            if True:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)

       
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1



        results = evaluate(model, tokenizer)
        print("Evaluate Accuracy", results)
        best_acc = max(best_acc, results['qqp'])
        print("Best result:", best_acc)    
                  
    model_path = "hubert.pth"

    calibration_config = quantizer.export_model(model_path)
    results = evaluate(model, tokenizer)
    print("Evaluate Accuracy", results)
    print("best_acc", best_acc)

    return global_step, tr_loss / global_step

device = torch.device('cuda')
config = torch.load('Coarse_bert_config')
dummy_input = torch.load('dummy_input.pth')
data = (dummy_input['input_ids'].to(device), dummy_input['attention_mask'].to(device), dummy_input['token_type_ids'].to(device))
norm_model = BertForSequenceClassification(config=config).to(device)
head_prune_cfg = torch.load('head_prune_cfg')
norm_model.prune_heads(head_prune_cfg)
norm_model.load_state_dict(torch.load('/data/znx/SpargenCks/bert_coarse_cks/nni_weight.pth') )
task_name = "qqp"
token = BertTokenizer.from_pretrained('/data/znx/SpargenCks/bert_coarse_cks/token_pretrain/checkpoint-220000')
acc = evaluate(norm_model, token)
print(f"original accuracy is {acc}")

mask_path = "/data/znx/SpargenCks/bert_coarse_onnx_with_tesa_mixed_fp32/new_sparsity_mask.pth"
mask_dict = torch.load(mask_path)

non_zero_mask_path = "/data/znx/SpargenCks/bert_coarse_onnx_with_tesa_mixed_fp32/fp32_mask.pth"
nonzero_mask_dict = torch.load(non_zero_mask_path)

print("Begin to assign mask to quantize wrapper")
for _, module in norm_model.named_modules():
    if hasattr(module, 'name'):
        if module.name in mask_dict.keys():
            name = module.name
            module.mask_weight = mask_dict[name]['weight']
            module.mask_bias = mask_dict[name]['bias']
            module.mask_non_zero = nonzero_mask_dict[name]['weight']
print("Finish assign mask to quantize wrapper")

configure_list = [{
    'quant_types': ['weight', 'output'],
    'quant_bits': {
        'input':8,
        'weight': 8,
        'output':8
    }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
    'op_types':['Conv2d', 'Linear'],
    'quant_start_step': 1000
}]

train_dataset = load_and_cache_examples("qqp", token, evaluate=False)

optimizer, scheduler = get_optimizer_scheduler(norm_model, train_dataset)

quantizer = QAT_Quantizer(norm_model, configure_list, optimizer)
quantizer.compress()



train(train_dataset, norm_model, token, optimizer, scheduler)

import pdb; pdb.set_trace()


export AZCOPY_CRED_TYPE="Anonymous"; azcopy copy "/data/znx/MLPruning/inference/related_file"  "https://nennistorage.blob.core.windows.net/nni/v-linbin/MLPruning?sv=2020-08-04&ss=b&srt=sco&sp=rwdlacx&se=2022-07-01T14:41:00Z&st=2021-09-09T06:41:00Z&spr=https&sig=vVyyBj9q6RSjNODflKjUCwh3TW1%2B1N9W5sp%2BdcATVto%3D"  --recursive; export AZCOPY_CRED_TYPE="";