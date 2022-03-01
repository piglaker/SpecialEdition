#The final version
d=`date`
datetime=${d// /-}

#dataset="ctc2021"
#model_name="CPT_NLG"
#use_extra_dataset=True


dataset="sighan"
model_name="CPT_NLG"
use_extra_dataset=False

epoch=10
batch_size=128

eval_dataset="15"

name=$model_name"_of_extra"$use_extra_dataset"_eval"$eval_dataset"_epoch"$epoch"_bs"$batch_size

# echo "cat logs/$dataset/$name.log & gpustat" > check_stat.sh

CUDA_VISIBLE_DEVICES=6  nohup python proto_model.py \
    --seed 233 \
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
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --dataloader_pin_memory True \
    --metric_for_best_model F1_score \
    --dataset $dataset \
    --eval_dataset $eval_dataset \
    --learning_rate 7e-5 \
    --warmup_steps 5000 \
    --eval_steps 2000 \
    --model_name $model_name \
    --use_extra_dataset $use_extra_dataset \
> logs/$dataset/$name.log 2>&1 &