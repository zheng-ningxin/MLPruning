T_OUTPUT_PATH=result/qqp_partial/0.5/checkpoint-132000/ # used to get row/head pruning
bs=1
seq=128
OUTPUT_PATH=result/qqp_full/16_16/0.3/checkpoint-220000/
export CUDA_VISIBLE_DEVICES=1; python masked_bert_parameter_count.py \
                            --model_type masked_bert \
                            --model_name_or_path ${T_OUTPUT_PATH}  \
                            --block_path ${OUTPUT_PATH} \
                            --per_gpu_train_batch_size ${bs} \
                            --max_seq_length ${seq} \
                            --pruning_method topK \
                            --block_cols 16 --block_rows 16 \
                            --head_pruning
