source $TREC_VENV/bin/activate
export DATA=$TREC/data2

source $CUTTSUM/events.sh
source $CUTTSUM/models.sh


for query_event in "${TS_EVENTS_2013[@]}"; do
  
  IFS='|'; set ${query_event};
  event=$2 
  
  for model in "${TS_MODELS_2013[@]}"; do
    for dim in "100"; do
      for temp_mode in temp,max,100 temp,agg,100 no-temp,ign,NaN ; do 
        IFS=","; set $temp_mode; 
        use_temp=$1
        pen_mode=$2
        scale=$3
SYS_FILE="${DATA}/summarizer-output/ap/${event}/scale.${scale}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}/updates.txt" 
MOD_FILE=$DATA/summarizer-output/models/${event}/updates.txt
OFILE=$DATA/evaluation/auto-trec/ap/${event}/scale.${scale}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}/updates.txt

if [ ! -f $SYS_FILE ] || [ ! -f $MOD_FILE ]; then
    continue
fi

echo "$SYS_FILE VS $MOD_FILE" 
python -u $CUTTSUM/eval-scripts/auto_trec_eval.py \
    -s $SYS_FILE \
    -m $MOD_FILE \
    --sent-sim-model $DATA/sentence-sim/${model}_spl_model/model_${dim}.p \
    --sent-sim-vocab $DATA/sentence-sim/${model}_spl_model/vocab.txt \
    -o $OFILE

      done
    done
  done
done

RANK="${DATA}/summarizer-output/rank"

for query_event in "${TS_EVENTS_2013[@]}"; do
  
  IFS='|'; set ${query_event};
  event=$2 
  
  for model in "${TS_MODELS_2013[@]}"; do
    for dim in "100"; do
      for topn in 1 5 10; do    
        for temp_mode in temp,max temp,agg no-temp,ign ; do 
            IFS=","; set $temp_mode; 
            use_temp=$1
            pen_mode=$2
SYS_FILE=${RANK}/${event}/top${topn}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}/updates.txt
MOD_FILE=$DATA/summarizer-output/models/${event}/updates.txt
OFILE=$DATA/evaluation/auto-trec/rank/${event}/top${topn}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}.tsv

if [ ! -f $SYS_FILE ] || [ ! -f $MOD_FILE ]; then
    continue
fi
 
python -u $CUTTSUM/eval-scripts/auto_trec_eval.py \
    -s $SYS_FILE \
    -m $MOD_FILE \
    --sent-sim-model $DATA/sentence-sim/${model}_spl_model/model_${dim}.p \
    --sent-sim-vocab $DATA/sentence-sim/${model}_spl_model/vocab.txt \
    -o $OFILE

        done
      done
    done
  done
done

