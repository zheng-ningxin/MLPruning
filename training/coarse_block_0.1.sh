T_OUTPUT_PATH=result/qqp_partial/coarse_0.3/checkpoint-220000/ # used to get row/head pruning
bs=1
seq=128
OUTPUT_PATH=result/qqp_full/32_32/0.1/checkpoint-220000/
export CUDA_VISIBLE_DEVICES=1; python masked_bert_parameter_count.py \
                            --model_type masked_bert \
                            --model_name_or_path ${T_OUTPUT_PATH}  \
                            --block_path ${OUTPUT_PATH} \
                            --per_gpu_train_batch_size ${bs} \
                            --max_seq_length ${seq} \
                            --pruning_method topK \
                            --block_cols 32 --block_rows 32 \
                            --head_pruning
