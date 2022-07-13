#The final version
d=`date`
datetime=${d// /-}

#task="ctc2021"
#dataset="ctc2021"
#model_name="CPT_NLG"
#use_extra_dataset=False

task="sighan"
dataset="sighan_raw"
model_name="Proto"

eval_dataset="15"

cl_weight=0
repeat_weight=0
copy_weight=0.01


fix_cls=False

if [ ! -d "./logs/$task/$dataset" ]; then
    mkdir ./logs/$task/$dataset
fi

#if [ "$model_name" == "Proto" ];then
#    fix_cls=True
#    name=${model_name}"_mask_cls_copy"${copy_weight}"_cl"${cl_weight}"_repeat"${repeat_weight}"_eval"${eval_dataset}"_epoch"${epoch}"_bs"${batch_size}
#else
#    name=${model_name}"_eval"${eval_dataset}"_epoch"${epoch}"_bs"${batch_size}
#fi

# echo "cat logs/$dataset/$name.log & gpustat" > check_stat.sh

# Automatically search for available gpus
gpu_memory=(`nvidia-smi -q -d Memory |grep -A4 GPU|grep Free  | awk -F" "    '{ print $3 }'`)

num_gpus=2

count=0

gtx1080=10240
gtx3090=20480

available_gpus=""

batch_size=48

for i in "${!gpu_memory[@]}";   
do   
    if [ "${gpu_memory[$i]}" -gt "$gtx3090" ]
    then
        available_gpus="$available_gpus$i,"
        let count+=1
    fi
    
    if [ "${gpu_memory[$i]}" -gt "$gtx3090" ]
    then
        batch_size=128
    fi

    if [ $count -ge $num_gpus ] 
    then
        break
    fi 
done  

if [ $count -lt $num_gpus ]
then 
    echo "Error: No enough GPUs!"
    exit
fi

epoch=10
batch_size=64

echo "Use GPUs: "${available_gpus}

pretrained_name=roberta # pretrain bert type: [ bert roberta macbert xlnet chinesebert electra albert roformer nezha ]

VALUE=1

head=BertForMaskedLM_CL #ConfusionCluster/3

output_dir=./tmp/${dataset}/${head}/${pretrained_name}

if [ "$model_name" == "Proto" ];then
    fix_cls=True
    name=${model_name}"_mask_cls_copy"${copy_weight}"_cl"${cl_weight}"_repeat"${repeat_weight}"_eval"${eval_dataset}"_epoch"${epoch}"_bs"${batch_size}
else
    name=${model_name}"_eval"${eval_dataset}"_epoch"${epoch}"_bs"${batch_size}
fi

log_path=logs/${task}/${dataset}/${head}/${model_name}/${pretrained_name}/${name}.log

#seed 153603 27 3472
#lr  5e-5 7e-5 6e-5

rm Recent_Note.log
echo "Log_path: "${log_path}
ln -s ${log_path} Recent_Note.log

CUDA_VISIBLE_DEVICES=${available_gpus} OMP_NUM_THREADS=${VALUE} torchrun --nproc_per_node=${num_gpus} --master_port 6500 --nnodes=1 --node_rank=0 \
    main.py \
    --sharded_ddp zero_dp_2 \
    --seed 3472 \
    --do_train \
    --do_eval \
    --do_predict \
    --fp16 False \
    --disable_tqdm False \
    --dataloader_num_workers 0 \
    --output_dir ${output_dir} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --eval_accumulation_steps 1 \
    --num_train_epochs ${epoch} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --dataloader_pin_memory True \
    --metric_for_best_model F1_score \
    --dataset ${dataset} \
    --eval_dataset ${eval_dataset} \
    --learning_rate 6e-5 \
    --warmup_steps 2500 \
    --eval_steps 1000 \
    --save_total_limit 1 \
    --model_name ${model_name} \
    --use_extra_dataset ${use_extra_dataset} \
    --fix_cls ${fix_cls} \
    --cl_weight ${cl_weight} \
    --repeat_weight ${repeat_weight} \
    --copy_weight ${copy_weight} \
    --num_gpus ${num_gpus} \
    --pretrained_name ${pretrained_name} \
    --log_path ${log_path} \