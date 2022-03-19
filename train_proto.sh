#The final version
d=`date`
datetime=${d// /-}

#task="ctc2021"
#dataset="ctc2021"
#model_name="CPT_NLG"
#use_extra_dataset=False

task="sighan"
dataset="sighan_enchanted"
model_name="Proto"

epoch=10
batch_size=128

eval_dataset="15"

cl_weight=0
repeat_weight=0
copy_weight=30

if [ ! -d "./logs/$task/$dataset" ]; then
    mkdir ./logs/$task/$dataset
fi

if [ "$model_name" == "Proto" ];then

    name=$model_name"_mask_cls_copy"$copy_weight"_cl"$cl_weight"_repeat"$repeat_weight"_eval"$eval_dataset"_epoch"$epoch"_bs"$batch_size
else
    name=$model_name"_eval"$eval_dataset"_epoch"$epoch"_bs"$batch_size
fi

# echo "cat logs/$dataset/$name.log & gpustat" > check_stat.sh

CUDA_VISIBLE_DEVICES=5 nohup python proto_model.py \
    --seed 153603 \
    --do_train \
    --do_eval \
    --do_predict \
    --fp16 True \
    --disable_tqdm False \
    --dataloader_num_workers 0 \
    --output_dir ./tmp/$dataset/$name \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --eval_accumulation_steps 1 \
    --num_train_epochs $epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --dataloader_pin_memory True \
    --metric_for_best_model F1_score \
    --dataset $dataset \
    --eval_dataset $eval_dataset \
    --learning_rate 7e-5 \
    --warmup_steps 10000 \
    --eval_steps 2000 \
    --save_total_limit 0 \
    --model_name $model_name \
    --use_extra_dataset $use_extra_dataset \
    --fix_cls False \
    --cl_weight $cl_weight \
    --repeat_weight $repeat_weight \
    --copy_weight $copy_weight \
> logs/$task/$dataset/$name.log 2>&1 &