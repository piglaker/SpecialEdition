dataset="sighan"
eval_dataset="15"
model_name="Dot"
epoch=10
batch_size=16

name="bert_Raw_base_ReaLiSe_metric.$model_name.$eval_dataset.epoch$epoch.bs$batch_size"

echo "cat logs/$dataset/$name.log & gpustat" > check_stat.sh

CUDA_VISIBLE_DEVICES=4 nohup python bert_MaskedLM.py \
    --do_train \
    --do_eval \
    --do_predict \
    --fp16 True \
    --disable_tqdm False \
    --dataloader_num_workers 0 \
    --output_dir ./tmp/$dataset/$name \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --eval_accumulation_steps 2 \
    --num_train_epochs $epoch \
    --evaluation_strategy steps \
    --save_strategy epoch \
    --dataloader_pin_memory True \
    --metric_for_best_model F1_score \
    --dataset $dataset \
    --eval_dataset $eval_dataset \
    --learning_rate 5e-5 \
    --warmup_steps 10000 \
    --eval_steps 2000 \
    --model_name $model_name \
> logs/$dataset/$name.log 2>&1 &
