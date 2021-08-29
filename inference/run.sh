
export PYTHONUNBUFFERED=1

OUTPUT_PATH=../training/result/qqp_partial/0.5/checkpoint-132000/
mkdir -p ${OUTPUT_PATH}
block_rows=16
block_cols=16
BLOCK_PATH=../training/result/qqp_full/${block_rows}_${block_cols}/0.3/checkpoint-220000/
batch_size=32
max_seq_length=512

export CUDA_VISIBLE_DEVICES=0; python masked_bert_inference.py --model_type masked_bert \
--model_name_or_path ${OUTPUT_PATH} --per_gpu_train_batch_size ${batch_size} \
--max_seq_length ${max_seq_length} --pruning_method topK \
--block_cols ${block_cols} --block_rows ${block_rows} \
--block_path ${BLOCK_PATH} --head_pruning
