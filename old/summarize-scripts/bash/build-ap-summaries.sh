source $CUTTSUM/events.sh
source $CUTTSUM/models.sh

mkdir -p $TREC/logs/ap-summarizer
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
SAL_DAT="${DATA}/salience-data/sentence-sims/${event}/${model}_spl_${dim}.txt"
O_DIR="${DATA}/summarizer-output/ap/${event}/scale.${scale}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}" 
if [ ! -f $SAL_DAT ]; then
    continue
fi

if [ ! -d $O_DIR ]; then
  echo "Summarizing ${event} by AP: scale.${scale}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}"
    python -u $CUTTSUM/summarize-scripts/python/ap_summarizer.py \
      -b "${DATA}/summarizer-input-data/sentence-bow/${event}.tsv" \
      -l "${DATA}/salience-data/sentence-sims/${event}/${model}_spl_${dim}.txt" \
      -o $O_DIR \
      --$use_temp \
      -p ${pen_mode} \
      &> "$TREC/logs/ap-summarizer/${event}.scale.${scale}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}" &

else
  echo "Skipping:" \
       "Summarizing ${event} by AP: scale.${scale}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}"
fi
      done
    done
  done
done

wait
