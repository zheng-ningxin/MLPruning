CUDA_VISIBLE_DEVICES=1 python evaluate_finegrained.py --model_name_or_path ../training/result/qqp_partial/1.0/checkpoint-113500/  --output_dir ./tmp_re --task_name qqp --model_type masked_bert 
