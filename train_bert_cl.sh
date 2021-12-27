dataset="sighan"
epoch=10
batch_size=64

name="bert_MaskedLM_CL_ReaLiSe_metric_warmup_test.epoch$epoch.bs$batch_size"

echo "cat logs/$dataset/$name.log & gpustat" > check_stat.sh
#default lr 5e-5
CUDA_VISIBLE_DEVICES=4 nohup python bert_MaskedLM_CL.py \
    --do_train \
    --do_eval \
    --do_predict \
    --fp16 True \
    --disable_tqdm False \
    --learning_rate 5e-5 \
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
    --warmup_steps 10000 \
    --eval_steps 1000 \
> logs/$dataset/$name.log 2>&1 &
