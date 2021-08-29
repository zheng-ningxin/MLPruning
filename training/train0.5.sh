export PYTHONUNBUFFERED=1

threshold=0.5

teacher_path=result/qqp_partial/1.0/checkpoint-110000/ 

OUTPUT_PATH=result/qqp_partial/${threshold}

mkdir -p ${OUTPUT_PATH}

export CUDA_VISIBLE_DEVICES=1; python masked_run_glue.py --output_dir ${OUTPUT_PATH} --data_dir ../../data-bin/glue_data/QQP \
--task_name qqp --do_train --do_eval --do_lower_case --model_type masked_bert \
--model_name_or_path bert-base-uncased --per_gpu_train_batch_size 32 --overwrite_output_dir \
--warmup_steps 11000 --num_train_epochs 20 --max_seq_length 128 \
--learning_rate 3e-05 --mask_scores_learning_rate 1e-2 --evaluate_during_training \
--logging_steps 11000 --save_steps 11000 \
--teacher_type bert --teacher_name_or_path  ${teacher_path} \
--fp16 --final_threshold ${threshold} --head_pruning --final_lambda 20000 --pruning_method topK \
--mask_init constant --mask_scale 0. | tee -a ${OUTPUT_PATH}/training_log.txt 
