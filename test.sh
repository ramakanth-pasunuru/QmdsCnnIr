# Required environment variables:
# DATASET: choose the dataset (CNNDM / WIKI)
# MODEL_TYPE: choose the type of model (hier, he, order, query, heq, heo, hero)

BATCH_SIZE=8000
VISIBLE_GPUS="0"
GPU_RANKS="0"
WORLD_SIZE=1
MAX_SAMPLES=100000

EXTRA=""

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
        TRUNC_TGT_NTOKEN=120
        TRUNC_SRC_NTOKEN=200
        TRUNC_SRC_NBLOCK=8
        MAX_LENGTH=250
        MIN_LENGTH=35
        EXTRA="-coverage_penalty summary -stepwise_penalty True -block_ngram_repeat 3"
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
        TRUNC_SRC_NBLOCK=40
        MAX_LENGTH=400
        MIN_LENGTH=200
        EXTRA="-alpha 0.4"
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
    -mode test \
    -batch_size $BATCH_SIZE \
    -trunc_tgt_ntoken $TRUNC_TGT_NTOKEN \
    -trunc_src_ntoken $TRUNC_SRC_NTOKEN \
    -trunc_src_nblock $TRUNC_SRC_NBLOCK \
    -visible_gpus $VISIBLE_GPUS \
    -gpu_ranks $GPU_RANKS \
    -world_size $WORLD_SIZE \
    -dataset $DATASET \
    -model_type $MODEL_TYPE \
    -query $QUERY \
    -max_samples $MAX_SAMPLES \
    -data_path $DATA_PATH \
    -vocab_path $VOCAB_PATH \
    -test_from $MODEL_PATH \
    -result_path $MODEL_PATH/outputs \
    -report_rouge \
    -max_length $MAX_LENGTH \
    -min_length $MIN_LENGTH \
    $EXTRA
