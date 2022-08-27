# My run.sh
d=`date`
datetime=${d// /-}

# Automatically search available gpus
echo "Searching available gpus..."

gpu_memory=(`nvidia-smi -q -d Memory |grep -A4 GPU|grep Free  | awk -F" "    '{ print $3 }'`)

NUM_GPUS=6

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

echo "Use GPUs: "${available_gpus}

#task="ctc2021"
#dataset="ctc2021"
#model_name="CPT_NLG"
#use_extra_dataset=False

TASK="sighan"
DATASET="sighan_ReaLiSe"
MODEL_NAME="Proto"

if [ ! -d "./logs/${TASK}/${DATASET}" ]; then
    mkdir ./logs/${TASK}/${DATASET}
fi

EVAL_DATASET="15"

if [ ! -n "$1" ] ;then
    CL_WEIGHT=0.0025 #0.005 0.05 
else
    CL_WEIGHT=$1
fi
#CL_WEIGHT=0.01 #0.005 0.05

REPEAT_WEIGHT=1 #1

if [ ! -n "$2" ] ;then
    MULTI_TASK_WEIGHT=0.005 #0.01 
else
    MULTI_TASK_WEIGHT=$2 #0.01 
fi
#MULTI_TASK_WEIGHT=0.1 #0.01 

MULTI_TASK=False

COPY_WEIGHT=0

FIX_CLS=False

PRETRAINED_NAME=macbert # pretrain bert type: [ bert roberta macbert xlnet chinesebert electra albert roformer nezha ]

EPOCH=20
STRATEGY=epoch
EVAL_STEPS=500
LEARNING_RATE=7e-5
WEIGHT_DECAY=0.02
WARMUP_STEPS=2500
CKPT_LIMIT=1

if [ ! -n "$3" ] ;then
    SEED=3471 #0.005 0.05 
else
    SEED=$3
fi

VALUE=1

HEAD=Proto # BertForMaskedLM_CL #ConfusionCluster/3 Proto CPT

if [ "${MODEL_NAME}" == "Proto" ];then
    fix_cls=True
    name=${MODEL_NAME}"_cls_copy"${COPY_WEIGHT}"_cl"${CL_WEIGHT}"_repeat"${REPEAT_WEIGHT}"_eval"${EVAL_DATASET}"_epoch"${EPOCH}"_bs"${BATCH_SIZE}"_seed"${SEED}"_multi_task"${MULTI_TASK}"_weight"${MULTI_TASK_WEIGHT}"_v2"
else
    name=${MODEL_NAME}"_eval"${EVAL_DATASET}"_epoch"${EPOCH}"_bs"${BATCH_SIZE}"_seed"${SEED}
fi

OUTPUT_DIR=./tmp/${DATASET}/${HEAD}/${PRETRAINED_NAME}/${name}

LOG_PATH=logs/${TASK}/${DATASET}/${HEAD}/${MODEL_NAME}/${PRETRAINED_NAME}/${name}.log

#seed 153603 27 3472
#lr  5e-5 7e-5 6e-5

rm Recent_Note.log
echo "LOG_PATH: "${LOG_PATH}
ln -s ${LOG_PATH} Recent_Note.log

CUDA_VISIBLE_DEVICES=${available_gpus} OMP_NUM_THREADS=${VALUE} CUDA_LAUNCH_BLOCKING=0 torchrun --nproc_per_node=${NUM_GPUS} --master_port 6501 --nnodes=1 --node_rank=0 \
    main.py \
    --sharded_ddp zero_dp_2 \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --fp16 True \
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
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_steps ${WARMUP_STEPS} \
    --save_steps ${EVAL_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --save_total_limit ${CKPT_LIMIT} \
    --model_name ${MODEL_NAME} \
    --use_extra_dataset ${use_extra_dataset} \
    --fix_cls ${FIX_CLS} \
    --multi_task ${MULTI_TASK} \
    --cl_weight ${CL_WEIGHT} \
    --repeat_weight ${REPEAT_WEIGHT} \
    --copy_weight ${COPY_WEIGHT} \
    --multi_task_weight ${MULTI_TASK_WEIGHT} \
    --num_gpus ${NUM_GPUS} \
    --pretrained_name ${PRETRAINED_NAME} \
    --log_path ${LOG_PATH} \
> note.log 2>&1

#cat tmp.log

echo "Finish training"
echo ${LOG_PATH}
