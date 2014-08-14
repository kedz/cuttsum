

if [ ! -d $DATA/models/salience-models/2012_Buenos_Aires_Rail_Disaster/accidents_100_spl_max ]; then
    echo "Training GP regression for 2012_Buenos_Aires_Rail_Disaster" \
    "lvec: accidents, 100, sim: max"
    mkdir -p $TREC/logs/salience-predictions/
    python -u $CUTTSUM/model-training-scripts/python/fit_salience_model.py \
        -m max \
        -s $DATA/salience-data/sentence-sims/2012_Buenos_Aires_Rail_Disaster/accidents_spl_100.txt \
        -f $DATA/salience-data/sentence-features/2012_Buenos_Aires_Rail_Disaster.txt \
        --model-dir $DATA/models/salience-models/2012_Buenos_Aires_Rail_Disaster/accidents_100_spl_max \
        --prediction-file $DATA/salience-data/predicted-sentence-sims/2012_Buenos_Aires_Rail_Disaster/accidents_100_spl_max.tsv \
        1>$TREC/logs/salience-predictions/2012_Buenos_Aires_Rail_Disaster.accidents_100_spl_max.log \
        2>$TREC/logs/salience-predictions/2012_Buenos_Aires_Rail_Disaster.accidents_100_spl_max.err &

else
    echo "Skipping: Training GP regression for 2012_Buenos_Aires_Rail_Disaster" \
    "lvec: accidents, 100, sim: max"
fi

if [ ! -d $DATA/models/salience-models/2012_Buenos_Aires_Rail_Disaster/accidents_100_spl_min ]; then
    echo "Training GP regression for 2012_Buenos_Aires_Rail_Disaster" \
    "lvec: accidents, 100, sim: min"
    mkdir -p $TREC/logs/salience-predictions/
    python -u $CUTTSUM/model-training-scripts/python/fit_salience_model.py \
        -m min \
        -s $DATA/salience-data/sentence-sims/2012_Buenos_Aires_Rail_Disaster/accidents_spl_100.txt \
        -f $DATA/salience-data/sentence-features/2012_Buenos_Aires_Rail_Disaster.txt \
        --model-dir $DATA/models/salience-models/2012_Buenos_Aires_Rail_Disaster/accidents_100_spl_min \
        --prediction-file $DATA/salience-data/predicted-sentence-sims/2012_Buenos_Aires_Rail_Disaster/accidents_100_spl_min.tsv \
        1>$TREC/logs/salience-predictions/2012_Buenos_Aires_Rail_Disaster.accidents_100_spl_min.log \
        2>$TREC/logs/salience-predictions/2012_Buenos_Aires_Rail_Disaster.accidents_100_spl_min.err &

else
    echo "Skipping: Training GP regression for 2012_Buenos_Aires_Rail_Disaster" \
    "lvec: accidents, 100, sim: min"
fi

if [ ! -d $DATA/models/salience-models/2012_Buenos_Aires_Rail_Disaster/accidents_100_spl_avg ]; then
    echo "Training GP regression for 2012_Buenos_Aires_Rail_Disaster" \
    "lvec: accidents, 100, sim: avg"
    mkdir -p $TREC/logs/salience-predictions/
    python -u $CUTTSUM/model-training-scripts/python/fit_salience_model.py \
        -m avg \
        -s $DATA/salience-data/sentence-sims/2012_Buenos_Aires_Rail_Disaster/accidents_spl_100.txt \
        -f $DATA/salience-data/sentence-features/2012_Buenos_Aires_Rail_Disaster.txt \
        --model-dir $DATA/models/salience-models/2012_Buenos_Aires_Rail_Disaster/accidents_100_spl_avg \
        --prediction-file $DATA/salience-data/predicted-sentence-sims/2012_Buenos_Aires_Rail_Disaster/accidents_100_spl_avg.tsv \
        1>$TREC/logs/salience-predictions/2012_Buenos_Aires_Rail_Disaster.accidents_100_spl_avg.log \
        2>$TREC/logs/salience-predictions/2012_Buenos_Aires_Rail_Disaster.accidents_100_spl_avg.err &

else
    echo "Skipping: Training GP regression for 2012_Buenos_Aires_Rail_Disaster" \
    "lvec: accidents, 100, sim: avg"
fi

wait
