import argparse
import glob
import json
import logging

import os
import random
import math

import onnx
import onnx.numpy_helper

import numpy as np
import torch
# from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from nni.compression.pytorch.utils import get_module_by_name
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
import nni
import torch
import sys
import os
from nni.algorithms.compression.pytorch.pruning import LevelPruner

mask_path = './mask'
weight_path = './weight'
head_pruner_cfg = torch.load('head_prune_cfg')
task_name = "qqp"
model_name_or_path = '../training/result/qqp_partial/1.0/checkpoint-220000/'
data_dir = './QQP'
max_seq_length= 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_epochs = 50

def original_main():
    output_mode = output_modes[task_name]
    tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-QQP')
    config = BertConfig.from_pretrained('textattack/bert-base-uncased-QQP')
    model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-QQP', config=config)

    model = model.to(device)
    cfg_list = [{'op_types':['Linear'], 'sparsity':0.95}]
    pruner = LevelPruner(model, cfg_list)
    pruner.compress()
    print(evaluate(model, tokenizer))
    train_dataset = load_and_cache_examples("qqp", tokenizer, evaluate=False)
    train(train_dataset, model, tokenizer)
    import pdb; pdb.set_trace()


class LayernameModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_bits) -> None:
        """
        Parameters
        ----------
        module : torch.nn.Module
            Layer module of pytorch model
        module_bits : int
            Bits width setting for module
        """
        super().__init__()
        self.module = module
        self.module_bits = module_bits

    def forward(self, inputs):
        inputs = inputs*self.module_bits
        inputs = self.module(inputs)
        return inputs

def _setattr(model, name, module):
    """
    Parameters
    ----------
    model : pytorch model
        The model to speed up by quantization
    name : str
        name of pytorch module
    module : torch.nn.Module
        Layer module of pytorch model
    """
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

def unwrapper(model_onnx, index2name, config):
    """
    Fill onnx config and remove wrapper node in onnx

    Parameters
    ----------
    model_onnx : onnx model
        Onnx model which is converted from pytorch model
    index2name : dict
        Dictionary of layer index and name
    config : dict
        Config recording name of layers and calibration parameters

    Returns
    -------
    onnx model
        Onnx model which is converted from pytorch model
    dict
        The configuration of onnx model layers and calibration parameters
    """
    # Support Gemm, Conv, Relu, Clip(Relu6) and Maxpool
    support_op = ['Gemm', 'Conv', 'MatM']
    op_names = []
    idx = 0
    onnx_config = {}
    while idx < len(model_onnx.graph.node):
        nd = model_onnx.graph.node[idx]
        op_names.append(nd.name)
        if nd.name[0:4] in support_op and  idx > 1:
            # Grad constant node and multiply node
            const_nd = model_onnx.graph.node[idx-2]
            mul_nd = model_onnx.graph.node[idx-1]

            if const_nd.name[0:8] != "Constant" or mul_nd.name[0:3] != "Mul":
                idx += 1
                continue

            # Get index number which is transferred by constant node
            index = int(onnx.numpy_helper.to_array(const_nd.attribute[0].t))
            if index != -1:
                name = index2name[index]
                onnx_config[nd.name] = config[name]
            nd.input[0] = mul_nd.input[0]
            # Remove constant node and multiply node
            model_onnx.graph.node.remove(const_nd)
            model_onnx.graph.node.remove(mul_nd)
            idx = idx-2
        idx = idx+1
    return model_onnx, onnx_config

def torch_to_onnx(model, config, model_path, input_names=["actual_input_1"], output_names=["output1"]):
    """
    Convert torch model to onnx model and get layer bits config of onnx model.

    Parameters
    ----------
    model : pytorch model
        The model to speed up by quantization
    config : dict
        Config recording bits number and name of layers
    input_shape : tuple
        The input shape of model, shall pass it to torch.onnx.export
    model_path : str
        The path user want to store onnx model which is converted from pytorch model
    input_names : list
        Input name of onnx model providing for torch.onnx.export to generate onnx model
    output_name : list
        Output name of onnx model providing for torch.onnx.export to generate onnx model

    Returns
    -------
    onnx model
        Onnx model which is converted from pytorch model
    dict
        The configuration of onnx model layers and calibration parameters
    """
    # Support Gemm, Conv, Relu, Clip(Relu6) and MaxPool
    support_op = [torch.nn.Conv2d, torch.nn.Linear]
    # Transfer bits number to onnx layer by using wrapper
    index2name = {}
    name2index = {}
    if config is not None:
        for i, name in enumerate(config.keys()):
            index2name[i] = name
            name2index[name] = i
    for name, module in model.named_modules():
        if config is not None and name in config:
            assert type(module) in support_op
            wrapper_module = LayernameModuleWrapper(module, name2index[name])
            _setattr(model, name, wrapper_module)
        elif type(module) in support_op:
            wrapper_module = LayernameModuleWrapper(module, -1)
            _setattr(model, name, wrapper_module)
    # Convert torch model to onnx model and save it in model_path

    # dummy input
    data = torch.load('dummy_input.pth')
    dummy_input = (data['input_ids'], data['attention_mask'], data['token_type_ids'])
    # dummy_input

    model.to('cpu')
    torch.onnx.export(model, dummy_input, model_path, opset_version=10)

    # Load onnx model
    model_onnx = onnx.load(model_path)
    model_onnx, onnx_config = unwrapper(model_onnx, index2name, config)
    onnx.save(model_onnx, model_path)

    onnx.checker.check_model(model_onnx)
    return model_onnx, onnx_config

def test_mapping():
    output_mode = output_modes[task_name]
    tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-QQP')
    config = BertConfig.from_pretrained('textattack/bert-base-uncased-QQP')
    model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-QQP', config=config)

    model = model.to(device)

    support_torch_op = [torch.nn.Conv2d, torch.nn.Linear]
    index_config = {}
    idx = 0
    for name, module in model.named_modules():
        if type(module) in support_torch_op:
            index_config[name] = idx
            idx += 1
    
    model_path = "bert.pth"
    print(index_config)
    model_onnx, onnx_config = torch_to_onnx(model, index_config, model_path)
    print("onnx_config: ", onnx_config)

if __name__ == '__main__':
    test_mapping()

