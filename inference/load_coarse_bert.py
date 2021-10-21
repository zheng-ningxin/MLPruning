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
import nni
import torch
import sys
import os
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from nni_finetune import *


device = torch.device('cuda')
config = torch.load('Coarse_bert_config')
dummy_input = torch.load('dummy_input.pth')
data = (dummy_input['input_ids'].to(device), dummy_input['attention_mask'].to(device), dummy_input['token_type_ids'].to(device))
norm_model = BertForSequenceClassification(config=config).to(device)
head_prune_cfg = torch.load('head_prune_cfg')
norm_model.prune_heads(head_pruner_cfg)
norm_model.load_state_dict(torch.load('/data/znx/SpargenCks/bert_coarse_cks/nni_weight.pth') )
task_name = "qqp"
token = BertTokenizer.from_pretrained('/data/znx/SpargenCks/bert_coarse_cks/token_pretrain/checkpoint-220000')
acc = evaluate(norm_model, token)

import pdb; pdb.set_trace()
