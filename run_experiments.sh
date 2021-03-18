# Required environment variables:
# DATASET: choose the dataset (CNNDM / WIKI)
# MODEL_TYPE: choose the type of model (hier, he, order, query, heq, heo, hero)

BATCH_SIZE=8000
SEED=666
TRAIN_STEPS=500000
SAVE_CHECKPOINT_STEPS=5000
REPORT_EVERY=100
VISIBLE_GPUS="0,1,2,3"
GPU_RANKS="0,1,2,3"
WORLD_SIZE=4
ACCUM_COUNT=2
DROPOUT=0.1
LABEL_SMOOTHING=0.1
INTER_LAYERS="6,7"
INTER_HEADS=8
LR=1
MAX_SAMPLES=500

case $MODEL_TYPE in
    query|heq|hero)
        QUERY=True
        ;;
    hier|he|order|heo)
        QUERY=False
        ;;
    *)
        echo "Invalid option: ${MODEL_TYPE}"
        ;;
esac

case $DATASET in
    CNNDM)
        TRUNC_TGT_NTOKEN=100
        TRUNC_SRC_NTOKEN=200
        TRUNC_SRC_NBLOCK=8
        if [ $QUERY == "False" ]; then
            DATA_FOLDER_NAME=pytorch_qmdscnn
        else
            DATA_FOLDER_NAME=pytorch_qmdscnn_query
        fi
        if [ -z ${DATA_PATH+x} ]; then 
            DATA_PATH="data/qmdscnn/${DATA_FOLDER_NAME}/CNNDM"
        fi
        if [ -z ${VOCAB_PATH+x} ]; then 
            VOCAB_PATH="data/qmdscnn/${DATA_FOLDER_NAME}/spm.model"
        fi
        ;;
    WIKI)
        TRUNC_TGT_NTOKEN=400
        TRUNC_SRC_NTOKEN=100
        TRUNC_SRC_NBLOCK=24
        if [ $QUERY == "False" ]; then
            DATA_FOLDER_NAME=ranked_wiki_b40
        else
            DATA_FOLDER_NAME=ranked_wiki_b40_query
        fi
        if [ -z ${DATA_PATH+x} ]; then 
            DATA_PATH="data/wikisum/${DATA_FOLDER_NAME}/WIKI"
        fi
        if [ -z ${VOCAB_PATH+x} ]; then 
            VOCAB_PATH="data/wikisum/${DATA_FOLDER_NAME}/spm9998_3.model"
        fi
        ;;
    *)
        echo "Invalid option: ${DATASET}"

esac

# If model path not set
if [ -z ${MODEL_PATH+x} ]; then
    MODEL_PATH="results/model-${DATASET}-${MODEL_TYPE}"
fi


python src/train_abstractive.py \
    -mode train \
    -batch_size $BATCH_SIZE \
    -seed $SEED \
    -train_steps $TRAIN_STEPS \
    -save_checkpoint_steps $SAVE_CHECKPOINT_STEPS \
    -report_every $REPORT_EVERY \
    -trunc_tgt_ntoken $TRUNC_TGT_NTOKEN \
    -trunc_src_ntoken $TRUNC_SRC_NTOKEN \
    -trunc_src_nblock $TRUNC_SRC_NBLOCK \
    -visible_gpus $VISIBLE_GPUS \
    -gpu_ranks $GPU_RANKS \
    -world_size $WORLD_SIZE \
    -accum_count $ACCUM_COUNT \
    -lr $LR \
    -dec_dropout $DROPOUT \
    -enc_dropout $DROPOUT \
    -label_smoothing $LABEL_SMOOTHING \
    -inter_layers $INTER_LAYERS \
    -inter_heads $INTER_HEADS \
    -hier \
    -dataset $DATASET \
    -model_type $MODEL_TYPE \
    -query $QUERY \
    -max_samples $MAX_SAMPLES \
    -data_path $DATA_PATH \
    -vocab_path $VOCAB_PATH \
    -model_path $MODEL_PATH \
    -result_path $MODEL_PATH/outputs \

