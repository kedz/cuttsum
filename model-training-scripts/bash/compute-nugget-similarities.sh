source $CUTTSUM/events.sh

 

for dim in 1000 500 100; do
  for mver in dpl spl; do

    for query_event in "${TS_EVENTS_2013[@]}"; do
  
IFS='|'; set ${query_event};
query=$1
event=$2 

if [ ! -f $DATA/salience-data/sentence-sims/${event}/all_${mver}_${dim}.txt ]; then
    echo "Computing ${event}/all_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/${event}
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t ${query} \
        -r $DATA/relevant_chunks/filter2/${event} \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/all_events_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/all_events_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/${event}/all_events_${mver}_${dim}.txt \
        &>$TREC/logs/compute-sent-sims/${event}/all_events_${mver}_${dim}.log &

else
    echo "Skipping: Computing ${event}/all_events_${mver}_${dim}.txt..."
fi

      done
      wait

if [ ! -f $DATA/salience-data/sentence-sims/2012_Buenos_Aires_Rail_Disaster/accidents_${mver}_${dim}.txt ]; then
    echo "Computing 2012_Buenos_Aires_Rail_Disaster/accidents_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/2012_Buenos_Aires_Rail_Disaster
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t "2012 Buenos Aires Rail Disaster" \
        -r $DATA/relevant_chunks/filter2/2012_Buenos_Aires_Rail_Disaster \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/accidents_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/accidents_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/2012_Buenos_Aires_Rail_Disaster/accidents_${mver}_${dim}.txt \
        1>$TREC/logs/compute-sent-sims/2012_Buenos_Aires_Rail_Disaster/accidents_${mver}_${dim}.log \
        2>$TREC/logs/compute-sent-sims/2012_Buenos_Aires_Rail_Disaster/accidents_${mver}_${dim}.err & 
else
    echo "Skipping: Computing 2012_Buenos_Aires_Rail_Disaster/accidents_${mver}_${dim}.txt..."
fi

if [ ! -f $DATA/salience-data/sentence-sims/2012_Pakistan_garment_factory_fires/accidents_${mver}_${dim}.txt ]; then
    echo "Computing 2012_Pakistan_garment_factory_fires/accidents_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/2012_Pakistan_garment_factory_fires
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t "2012 Pakistan garment factory fires" \
        -r $DATA/relevant_chunks/filter2/2012_Pakistan_garment_factory_fires \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/accidents_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/accidents_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/2012_Pakistan_garment_factory_fires/accidents_${mver}_${dim}.txt \
        1>$TREC/logs/compute-sent-sims/2012_Pakistan_garment_factory_fires/accidents_${mver}_${dim}.log \
        2>$TREC/logs/compute-sent-sims/2012_Pakistan_garment_factory_fires/accidents_${mver}_${dim}.err & 
else
    echo "Skipping: Computing 2012_Pakistan_garment_factory_fires/accidents_${mver}_${dim}.txt..."
fi

if [ ! -f $DATA/salience-data/sentence-sims/2012_Aurora_shooting/terrshoot_${mver}_${dim}.txt ]; then
    echo "Computing 2012_Aurora_shooting/terrshoot_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/2012_Aurora_shooting
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t "2012 Aurora shooting" \
        -r $DATA/relevant_chunks/filter2/2012_Aurora_shooting \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/terrshoot_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/terrshoot_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/2012_Aurora_shooting/terrshoot_${mver}_${dim}.txt \
        1>$TREC/logs/compute-sent-sims/2012_Aurora_shooting/terrshoot_${mver}_${dim}.log \
        2>$TREC/logs/compute-sent-sims/2012_Aurora_shooting/terrshoot_${mver}_${dim}.err & 
else
    echo "Skipping: Computing 2012_Aurora_shooting/terrshoot_${mver}_${dim}.txt..."
fi

if [ ! -f $DATA/salience-data/sentence-sims/Wisconsin_Sikh_temple_shooting/terrshoot_${mver}_${dim}.txt ]; then
    echo "Computing Wisconsin_Sikh_temple_shooting/terrshoot_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/Wisconsin_Sikh_temple_shooting
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t "Wisconsin Sikh temple shooting" \
        -r $DATA/relevant_chunks/filter2/Wisconsin_Sikh_temple_shooting \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/terrshoot_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/terrshoot_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/Wisconsin_Sikh_temple_shooting/terrshoot_${mver}_${dim}.txt \
        1>$TREC/logs/compute-sent-sims/Wisconsin_Sikh_temple_shooting/terrshoot_${mver}_${dim}.log \
        2>$TREC/logs/compute-sent-sims/Wisconsin_Sikh_temple_shooting/terrshoot_${mver}_${dim}.err & 
else
    echo "Skipping: Computing Wisconsin_Sikh_temple_shooting/terrshoot_${mver}_${dim}.txt..."
fi

if [ ! -f $DATA/salience-data/sentence-sims/Hurricane_Isaac_2012/weather_${mver}_${dim}.txt ]; then
    echo "Computing Hurricane_Isaac_2012/weather_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/Hurricane_Isaac_2012
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t "Hurricane Isaac (2012)" \
        -r $DATA/relevant_chunks/filter2/Hurricane_Isaac_2012 \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/weather_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/weather_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/Hurricane_Isaac_2012/weather_${mver}_${dim}.txt \
        1>$TREC/logs/compute-sent-sims/Hurricane_Isaac_2012/weather_${mver}_${dim}.log \
        2>$TREC/logs/compute-sent-sims/Hurricane_Isaac_2012/weather_${mver}_${dim}.err & 
else
    echo "Skipping: Computing Hurricane_Isaac_2012/weather_${mver}_${dim}.txt..."
fi

if [ ! -f $DATA/salience-data/sentence-sims/Hurricane_Sandy/weather_${mver}_${dim}.txt ]; then
    echo "Computing Hurricane_Sandy/weather_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/Hurricane_Sandy
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t "Hurricane Sandy" \
        -r $DATA/relevant_chunks/filter2/Hurricane_Sandy \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/weather_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/weather_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/Hurricane_Sandy/weather_${mver}_${dim}.txt \
        1>$TREC/logs/compute-sent-sims/Hurricane_Sandy/weather_${mver}_${dim}.log \
        2>$TREC/logs/compute-sent-sims/Hurricane_Sandy/weather_${mver}_${dim}.err & 
else
    echo "Skipping: Computing Hurricane_Sandy/weather_${mver}_${dim}.txt..."
fi

if [ ! -f $DATA/salience-data/sentence-sims/June_2012_North_American_derecho/weather_${mver}_${dim}.txt ]; then
    echo "Computing June_2012_North_American_derecho/weather_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/June_2012_North_American_derecho
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t "June 2012 North American derecho" \
        -r $DATA/relevant_chunks/filter2/June_2012_North_American_derecho \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/weather_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/weather_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/June_2012_North_American_derecho/weather_${mver}_${dim}.txt \
        1>$TREC/logs/compute-sent-sims/June_2012_North_American_derecho/weather_${mver}_${dim}.log \
        2>$TREC/logs/compute-sent-sims/June_2012_North_American_derecho/weather_${mver}_${dim}.err & 
else
    echo "Skipping: Computing June_2012_North_American_derecho/weather_${mver}_${dim}.txt..."
fi

if [ ! -f $DATA/salience-data/sentence-sims/Typhoon_Bopha/weather_${mver}_${dim}.txt ]; then
    echo "Computing Typhoon_Bopha/weather_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/Typhoon_Bopha
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t "Typhoon Bopha" \
        -r $DATA/relevant_chunks/filter2/Typhoon_Bopha \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/weather_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/weather_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/Typhoon_Bopha/weather_${mver}_${dim}.txt \
        1>$TREC/logs/compute-sent-sims/Typhoon_Bopha/weather_${mver}_${dim}.log \
        2>$TREC/logs/compute-sent-sims/Typhoon_Bopha/weather_${mver}_${dim}.err & 
else
    echo "Skipping: Computing Typhoon_Bopha/weather_${mver}_${dim}.txt..."
fi

if [ ! -f $DATA/salience-data/sentence-sims/2012_Guatemala_earthquake/earthquakes_${mver}_${dim}.txt ]; then
    echo "Computing 2012_Guatemala_earthquake/earthquakes_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/2012_Guatemala_earthquake
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t "2012 Guatemala earthquake" \
        -r $DATA/relevant_chunks/filter2/2012_Guatemala_earthquake \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/earthquakes_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/earthquakes_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/2012_Guatemala_earthquake/earthquakes_${mver}_${dim}.txt \
        1>$TREC/logs/compute-sent-sims/2012_Guatemala_earthquake/earthquakes_${mver}_${dim}.log \
        2>$TREC/logs/compute-sent-sims/2012_Guatemala_earthquake/earthquakes_${mver}_${dim}.err & 
else
    echo "Skipping: Computing 2012_Guatemala_earthquake/earthquakes_${mver}_${dim}.txt..."
fi

if [ ! -f $DATA/salience-data/sentence-sims/2012_Tel_Aviv_bus_bombing/terrshoot_${mver}_${dim}.txt ]; then
    echo "Computing 2012_Tel_Aviv_bus_bombing/terrshoot_${mver}_${dim}.txt..."
    mkdir -p $TREC/logs/compute-sent-sims/2012_Tel_Aviv_bus_bombing
    python -u $CUTTSUM/model-training-scripts/python/compute_nugget_similarities.py \
        -t "2012 Tel Aviv bus bombing" \
        -r $DATA/relevant_chunks/filter2/2012_Tel_Aviv_bus_bombing \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -s $DATA/sentence-sim/terrshoot_${mver}_model/model_${dim}.p \
        -v $DATA/sentence-sim/terrshoot_${mver}_model/vocab.txt \
        -d ${dim} \
        -o $DATA/salience-data/sentence-sims/2012_Tel_Aviv_bus_bombing/terrshoot_${mver}_${dim}.txt \
        1>$TREC/logs/compute-sent-sims/2012_Tel_Aviv_bus_bombing/terrshoot_${mver}_${dim}.log \
        2>$TREC/logs/compute-sent-sims/2012_Tel_Aviv_bus_bombing/terrshoot_${mver}_${dim}.err & 
else
    echo "Skipping: Computing 2012_Tel_Aviv_bus_bombing/terrshoot_${mver}_${dim}.txt..."
fi

wait

  done 
done
