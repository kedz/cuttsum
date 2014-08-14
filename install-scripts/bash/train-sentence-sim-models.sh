#F1A="${LM_INPUT}/earthquakes_wiki_abstracts_spl.txt"
#F1B="${LM_INPUT}/earthquakes_wiki_abstracts_dpl.txt"
#F2A="${LM_INPUT}/weather_wiki_abstracts_spl.txt"
#F2B="${LM_INPUT}/weather_wiki_abstracts_dpl.txt"
#F3A="${LM_INPUT}/accidents_wiki_abstracts_spl.txt"
#F3B="${LM_INPUT}/accidents_wiki_abstracts_dpl.txt"
#F4A="${LM_INPUT}/terrshoot_wiki_abstracts_spl.txt"
#F4B="${LM_INPUT}/terrshoot_wiki_abstracts_dpl.txt"
#F5A="${LM_INPUT}/social_unrest_wiki_abstracts_spl.txt"
#F5B="${LM_INPUT}/social_unrest_wiki_abstracts_dpl.txt"
#FLISTA="$F1A $F2A $F3A $F4A $F5A"
#FLISTB="$F1B $F2B $F3B $F4B $F5B"
source $CUTTSUM/models.sh

if [ ! -d $TREC/logs/wtmf ]; then
    mkdir -p $TREC/logs/wtmf
fi


for mver in spl dpl; do
  FLIST=" "
  for model in "${TS_MODELS_2013[@]}"; do 
    FLIST="${FLIST} ${LM_INPUT}/${model}_wiki_abstracts_${mver}.txt"
  done

  for dim in 1000 500 100; do 

if [ ! -f  $DATA/sentence-sim/all_events_${mver}_model/model_${dim} ]; then
    echo "Training sentence sim for all events ${mver}..."
    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
    -i $FLIST -d $DATA/sentence-sim/all_events_${mver}_model/ \
    -v $LM_INPUT/all_abstracts_vocab.txt \
    --dims ${dim} \
    1>$TREC/logs/wtmf/all_events_${mver}_${dim}.log &
else
    echo "Skipping: Training sentence sim for all events ${mver} ${dim}..."
fi

  done
done

wait

for model in "${TS_MODELS_2013[@]}"; do 
  for mver in spl dpl; do
    for dim in 1000 500 100; do

if [ ! -f  $DATA/sentence-sim/${model}_${mver}_model/model_${dim} ]; then
    echo "Training sentence sim for ${model} ${mver}..."
    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
        -i ${LM_INPUT}/${model}_wiki_abstracts_${mver}.txt \
        -d $DATA/sentence-sim/${model}_${mver}_model/ \
        -v $LM_INPUT/${model}_wiki_abstracts_vocab.txt \
        --dims ${dim} \
        &>$TREC/logs/wtmf/${model}_${mver}_${dim}.log &

else
    echo "Skipping: Training sentence sim for ${model} ${mver}..."
fi

    done
    wait
  done
done


#exit
#if [ ! -d  $DATA/sentence-sim/all_events_spl_model/ ]; then
#    echo "Training sentence sim for all events spl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#    -i $FLISTA -d $DATA/sentence-sim/all_events_spl_model/ \
#    -v $LM_INPUT/all_abstracts_vocab.txt \
#    1>$TREC/logs/wtmf/all_events_spl.log \
#    2>$TREC/logs/wtmf/all_events_spl.err &
#else
#    echo "Skipping: Training sentence sim for all events spl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/all_events_dpl_model/ ]; then
#    echo "Training sentence sim for all events dpl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $FLISTB -d $DATA/sentence-sim/all_events_dpl_model/ \
#        -v $LM_INPUT/all_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/all_events_dpl.log \
#        2>$TREC/logs/wtmf/all_events_dpl.err &
#else
#    echo "Skipping: Training sentence sim for all events dpl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/earthquakes_spl_model/ ]; then
#    echo "Training sentence sim for earthquakes spl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $F1A -d $DATA/sentence-sim/earthquakes_spl_model/ \
#        -v $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/earthquakes_spl.log \
#        2>$TREC/logs/wtmf/earthquakes_spl.err &
#else
#    echo "Skipping: Training sentence sim for earthquakes spl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/earthquakes_dpl_model/ ]; then
#    echo "Training sentence sim for earthquakes dpl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $F1B -d $DATA/sentence-sim/earthquakes_dpl_model/ \
#        -v $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/earthquakes_dpl.log \
#        2>$TREC/logs/wtmf/earthquakes_dpl.err &
#else
#    echo "Skipping: Training sentence sim for earthquakes dpl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/weather_spl_model/ ]; then
#    echo "Training sentence sim for weather spl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $F2A -d $DATA/sentence-sim/weather_spl_model/ \
#        -v $LM_INPUT/weather_wiki_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/weather_spl.log \
#        2>$TREC/logs/wtmf/weather_spl.err &
#else
#    echo "Skipping: Training sentence sim for weather spl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/weather_dpl_model/ ]; then
#    echo "Training sentence sim for weather dpl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $F2B -d $DATA/sentence-sim/weather_dpl_model/ \
#        -v $LM_INPUT/weather_wiki_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/weather_dpl.log \
#        2>$TREC/logs/wtmf/weather_dpl.err &
#else
#    echo "Skipping: Training sentence sim for weather dpl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/accidents_spl_model/ ]; then
#    echo "Training sentence sim for accidents spl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $F3A -d $DATA/sentence-sim/accidents_spl_model/ \
#        -v $LM_INPUT/accidents_wiki_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/accidents_spl.log \
#        2>$TREC/logs/wtmf/accidents_spl.err &
#else
#    echo "Skipping: Training sentence sim for accidents spl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/accidents_dpl_model/ ]; then
#    echo "Training sentence sim for accidents dpl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $F3B -d $DATA/sentence-sim/accidents_dpl_model/ \
#        -v $LM_INPUT/accidents_wiki_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/accidents_dpl.log \
#        2>$TREC/logs/wtmf/accidents_dpl.err &
#else
#    echo "Skipping: Training sentence sim for accidents dpl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/terrshoot_spl_model/ ]; then
#    echo "Training sentence sim for terrshoot spl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $F4A -d $DATA/sentence-sim/terrshoot_spl_model/ \
#        -v $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/terrshoot_spl.log \
#        2>$TREC/logs/wtmf/terrshoot_spl.err &
#else
#    echo "Skipping: Training sentence sim for terrshoot spl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/terrshoot_dpl_model/ ]; then
#    echo "Training sentence sim for terrshoot dpl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $F4B -d $DATA/sentence-sim/terrshoot_dpl_model/ \
#        -v $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/terrshoot_dpl.log \
#        2>$TREC/logs/wtmf/terrshoot_dpl.err &
#else
#    echo "Skipping: Training sentence sim for terrshoot dpl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/social_unrest_spl_model/ ]; then
#    echo "Training sentence sim for social_unrest spl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $F5A -d $DATA/sentence-sim/social_unrest_spl_model/ \
#        -v $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/social_unrest_spl.log \
#        2>$TREC/logs/wtmf/social_unrest_spl.err &
#else
#    echo "Skipping: Training sentence sim for social_unrest spl..."
#fi
#
#if [ ! -d  $DATA/sentence-sim/social_unrest_dpl_model/ ]; then
#    echo "Training sentence sim for social_unrest dpl..."
#    python -u $CUTTSUM/install-scripts/python/train_sentence_similarity.py \
#        -i $F5B -d $DATA/sentence-sim/social_unrest_dpl_model/ \
#        -v $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt \
#        1>$TREC/logs/wtmf/social_unrest_dpl.log \
#        2>$TREC/logs/wtmf/social_unrest_dpl.err &
#else
#    echo "Skipping: Training sentence sim for social_unrest dpl..."
#fi
#
#wait
