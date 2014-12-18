source $CUTTSUM/events.sh
source $CUTTSUM/models.sh


mkdir -p $TREC/logs/rank-summarizer
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
SAL_DAT="${DATA}/salience-data/sentence-sims/${event}/${model}_spl_${dim}.txt"
O_DIR="${DATA}/summarizer-output/rank/${event}/top${topn}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}" 
if [ ! -f $SAL_DAT ]; then
    continue
fi

if [ ! -d $O_DIR ]; then
  echo "Summarizing ${event} by rank: top${topn}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}"
    python -u $CUTTSUM/summarize-scripts/python/rank_summarizer.py \
      -b "${DATA}/summarizer-input-data/sentence-bow/${event}.tsv" \
      -l "${DATA}/salience-data/sentence-sims/${event}/${model}_spl_${dim}.txt" \
      -o $O_DIR \
      --$use_temp \
      -p ${pen_mode} \
      -n $topn \
      &> "$TREC/logs/rank-summarizer/${event}.top${topn}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}" &

else
  echo "Skipping:" \
       "Summarizing ${event} by rank: top${topn}.${model}.gold.max.l${dim}.${use_temp}.${pen_mode}"
fi
        done
      done
    done
  done
done

wait

