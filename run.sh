#The final version
d=`date`
datetime=${d// /-}

#task="ctc2021"
#dataset="ctc2021"
#model_name="CPT_NLG"
#use_extra_dataset=False

TASK="sighan"
DATASET="sighan_raw"
MODEL_NAME="Proto"

EVAL_DATASET="15"

CL_WEIGHT=0
REPEAT_WEIGHT=1
COPY_WEIGHT=0.0005


FIX_CLS=False

if [ ! -d "./logs/${TASK}/${DATASET}" ]; then
    mkdir ./logs/${TASK}/${DATASET}
fi

#if [ "$model_name" == "Proto" ];then
#    fix_cls=True
#    name=${model_name}"_mask_cls_copy"${COPY_WEIGHT}"_cl"${CL_WEIGHT}"_repeat"${REPEAT_WEIGHT}"_eval"${eval_dataset}"_epoch"${epoch}"_bs"${batch_size}
#else
#    name=${model_name}"_eval"${eval_dataset}"_epoch"${epoch}"_bs"${batch_size}
#fi

# echo "cat logs/$dataset/$name.log & gpustat" > check_stat.sh

# Automatically search for available gpus
gpu_memory=(`nvidia-smi -q -d Memory |grep -A4 GPU|grep Free  | awk -F" "    '{ print $3 }'`)

NUM_GPUS=2

count=0

gtx1080=10240
gtx3090=20480

available_gpus=""

BATCH_SIZE=48

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

    if [ $count -ge $NUM_GPUS ] 
    then
        break
    fi 
done  

if [ $count -lt $NUM_GPUS ]
then 
    echo "Error: No enough GPUs!"
    exit
fi

EPOCH=10

echo "Use GPUs: "${available_gpus}

PRETRAINED_NAME=roberta # pretrain bert type: [ bert roberta macbert xlnet chinesebert electra albert roformer nezha ]

VALUE=1

HEAD=Proto # BertForMaskedLM_CL #ConfusionCluster/3 Proto CPT

OUTPUT_DIR=./tmp/${DATASET}/${HEAD}/${PRETRAINED_NAME}

if [ "${MODEL_NAME}" == "Proto" ];then
    fix_cls=True
    name=${MODEL_NAME}"_cls_copy"${COPY_WEIGHT}"_cl"${CL_WEIGHT}"_repeat"${REPEAT_WEIGHT}"_eval"${EVAL_DATASET}"_epoch"${EPOCH}"_bs"${BATCH_SIZE}
else
    name=${MODEL_NAME}"_eval"${EVAL_DATASET}"_epoch"${EPOCH}"_bs"${BATCH_SIZE}
fi

LOG_PATH=logs/${TASK}/${DATASET}/${HEAD}/${MODEL_NAME}/${PRETRAINED_NAME}/${name}.log

#seed 153603 27 3472
#lr  5e-5 7e-5 6e-5

rm Recent_Note.log
echo "LOG_PATH: "${LOG_PATH}
ln -s ${LOG_PATH} Recent_Note.log

STRATEGY=epoch
EVAL_STEPS=1000
LEARNING_RATE=6e-5
WARMUP_STEPS=2500
CKPT_LIMIT=0

SEED=3471

CUDA_VISIBLE_DEVICES=${available_gpus} OMP_NUM_THREADS=${VALUE} torchrun --nproc_per_node=${NUM_GPUS} --master_port 6500 --nnodes=1 --node_rank=0 \
    main.py \
    --sharded_ddp zero_dp_2 \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --fp16 False \
    --disable_tqdm False \
    --dataloader_num_workers 0 \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --eval_accumulation_steps 1 \
    --num_train_epochs ${EPOCH} \
    --evaluation_strategy ${STRATEGY} \
    --save_strategy ${STRATEGY} \
    --load_best_model_at_end \
    --dataloader_pin_memory True \
    --metric_for_best_model F1_score \
    --dataset ${DATASET} \
    --eval_dataset ${EVAL_DATASET} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --save_steps ${EVAL_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --save_total_limit ${CKPT_LIMIT} \
    --model_name ${MODEL_NAME} \
    --use_extra_dataset ${use_extra_dataset} \
    --fix_cls ${FIX_CLS} \
    --cl_weight ${CL_WEIGHT} \
    --repeat_weight ${REPEAT_WEIGHT} \
    --copy_weight ${COPY_WEIGHT} \
    --num_gpus ${NUM_GPUS} \
    --pretrained_name ${PRETRAINED_NAME} \
    --log_path ${LOG_PATH} 
#> tmp.log 2>&1

# cat tmp.log