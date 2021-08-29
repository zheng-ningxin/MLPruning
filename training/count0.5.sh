OUTPUT_PATH=result/qqp_partial/0.5/checkpoint-132000/
bs=1
seq=128
export CUDA_VISIBLE_DEVICES=1; python masked_bert_parameter_count.py \
                            --model_type masked_bert \
                            --model_name_or_path ${OUTPUT_PATH}  \
                            --per_gpu_train_batch_size ${bs} \
                            --max_seq_length ${seq} \
                            --pruning_method topK \
                            --block_cols -1 --block_rows -1 \
                            --head_pruning
