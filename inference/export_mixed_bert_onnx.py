from transformers.utils.dummy_pt_objects import get_polynomial_decay_schedule_with_warmup
from load_coarse_bert import *
from SparGen.Common.Utils import *
from nni.compression.pytorch.utils import get_module_by_name
new_nni_mask = torch.load('/data/znx/SpargenCks/bert_mixed_cks/new_sparsity_mask.pth')
fp32_mask = torch.load('/data/znx/SpargenCks/bert_mixed_cks/fp32_mask.pth')
for name in fp32_mask:
    fp_pos = fp32_mask[name]['weight'] > 0
    _, module = get_module_by_name(model, name)
    module.weight.data[fp_pos] = 0.002 # fake data now
export_tesa(model, data, 'bert_mixed_onnx_with_tesa', new_nni_mask)