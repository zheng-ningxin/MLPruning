CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name_or_path ../training/result/qqp_partial/coarse_0.3/checkpoint-220000/ --block_path ../training/result/qqp_full/32_32/0.1/checkpoint-220000/ --output_dir ./tmp_re --task_name qqp --model_type masked_bert --block_rows 32 --block_cols 32
