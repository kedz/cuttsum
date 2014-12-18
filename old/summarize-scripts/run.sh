source $TREC_VENV/bin/activate
export DATA=$TREC/data2

echo "BUIDLING MODEL SUMMARIES FROM NUGGET FILE"
echo "========================================="
bash $CUTTSUM/summarize-scripts/bash/build-model-summaries.sh
echo


echo "BUILDING BASELINE RANK SUMMARIES"
echo "================================"
bash $CUTTSUM/summarize-scripts/bash/build-rank-summaries.sh
echo

echo "BUILDING AP SUMMARIES"
echo "====================="
bash $CUTTSUM/summarize-scripts/bash/build-ap-summaries.sh
echo

exit

##if [ ! -f $DATA/summarizer-output/2012_Guatemala_earthquake/debug.max.lvecs100.txt ]; then
#    echo "Summarizing DEBUG NO-TEMP MAX EARTHQUAKES 100d-lvecs" \
#    "2012_Guatemala_earthquake"
#    mkdir -p $TREC/logs/summarizer/
    python -u $CUTTSUM/summarize-scripts/python/ap_summarizer.py \
        -b $DATA/summarizer-input-data/sentence-bow/2012_Guatemala_earthquake.tsv \
        -l $DATA/salience-data/sentence-sims/2012_Guatemala_earthquake/earthquakes_spl_100.txt \
        -o 2012_Guatemala_earthquake/debug.gold.max.lvecs100

exit
#        --energy-log $DATA/summarizer-output/energy-logs/2012_Guatemala_earthquake.debug.max.lvecs100.tsv \
#        -r $DATA/summarizer-output/rouge/2012_Guatemala_earthquake.debug.max.lvecs100 \
#        -c $DATA/summarizer-output/clusters/2012_Guatemala_earthquake.debug.max.lvecs100.txt \
#        1>$TREC/logs/summarizer/2012_Guatemala_earthquake.debug.max.lvecs100.log \
#        2>$TREC/logs/summarizer/2012_Guatemala_earthquake.debug.max.lvecs100.err &
#else
#    echo "Skipping: Summarizing DEBUG NO-TEMP MAX EARTHQUAKES 100d-lvecs" \
#    "2012_Guatemala_earthquake"
#fi

    python -u $CUTTSUM/summarize-scripts/temporal_summarizer.py \
        -b $DATA/summarizer-input-data/sentence-bow/2012_Pakistan_garment_factory_fires.tsv \
        -l $DATA/salience-data/sentence-sims/2012_Pakistan_garment_factory_fires/accidents_spl_100.txt \
        -o 2012_Pakistan_garment_factory_fires/debug.gold.max.lvecs100 \

#
##if [ ! -f $DATA/summarizer-output/2012_Buenos_Aires_Rail_Disaster/debug.max.lvecs100.txt ]; then
    echo "Summarizing DEBUG GOLD MAX ACCIDENTS 100d-lvecs" \
    "2012_Buenos_Aires_Rail_Disaster"
    python -u $CUTTSUM/summarize-scripts/temporal_summarizer.py \
        -b $DATA/summarizer-input-data/sentence-bow/2012_Buenos_Aires_Rail_Disaster.tsv \
        -l $DATA/salience-data/sentence-sims/2012_Buenos_Aires_Rail_Disaster/accidents_spl_100.txt \
        -o 2012_Buenos_Aires_Rail_Disaster/debug.gold.max.lvecs100 \
#        -p -20
#        &>2012_Buenos_Aires_Rail_Disaster.debug.gold.max.lvecs100.log &

#        &> 2012_Pakistan_garment_factory_fires.debug.gold.max.lvecs100.log &


#        --energy-log $DATA/summarizer-output/energy-logs/2012_Pakistan_garment_factory_fires.debug.max.lvecs100.tsv \
#        -r $DATA/summarizer-output/rouge/2012_Pakistan_garment_factory_fires.debug.max.lvecs100 \
#        -c $DATA/summarizer-output/clusters/2012_Pakistan_garment_factory_fires.debug.max.lvecs100.txt \
#        1>$TREC/logs/summarizer/2012_Pakistan_garment_factory_fires.debug.max.lvecs100.log \
#        2>$TREC/logs/summarizer/2012_Pakistan_garment_factory_fires.debug.max.lvecs100.err &



#        --energy-log energy-logs/2012_Buenos_Aires_Rail_Disaster.debug.max.lvecs100.tsv \
#        -r 2012_Buenos_Aires_Rail_Disaster.debug.max.lvecs100 \
#        -c 2012_Buenos_Aires_Rail_Disaster.debug.max.lvecs100.txt \


#        -o $DATA/summarizer-output/exemplars/2012_Buenos_Aires_Rail_Disaster/debug.gold.max.lvecs100.txt \
#        --energy-log $DATA/summarizer-output/energy-logs/2012_Buenos_Aires_Rail_Disaster.debug.max.lvecs100.tsv \
#        -r $DATA/summarizer-output/rouge/2012_Buenos_Aires_Rail_Disaster.debug.max.lvecs100 \
#        -c $DATA/summarizer-output/clusters/2012_Buenos_Aires_Rail_Disaster.debug.max.lvecs100.txt \
#        1>$TREC/logs/summarizer/2012_Buenos_Aires_Rail_Disaster.debug.max.lvecs100.log \
#        2>$TREC/logs/summarizer/2012_Buenos_Aires_Rail_Disaster.debug.max.lvecs100.err &
#
##else
##    echo "Skipping: Summarizing DEBUG NO-TEMP MAX ACCIDENTS 100d-lvecs" \
##    "2012_Buenos_Aires_Rail_Disaster"
##fi
#
#wait
#
##if [ ! -f $DATA/summarizer-output/2012_Pakistan_garment_factory_fires/debug.max.lvecs100.txt ]; then
#    echo "Summarizing DEBUG NO-TEMP MAX ACCIDENTS 100d-lvecs" \
#    "2012_Pakistan_garment_factory_fires"
#    mkdir -p $TREC/logs/summarizer/
#    python -u $CUTTSUM/summarize-scripts/temporal_summarizer.py \
#        -b $DATA/summarizer-input-data/sentence-bow/2012_Pakistan_garment_factory_fires.tsv \
#        -l $DATA/salience-data/sentence-sims/2012_Pakistan_garment_factory_fires/accidents_spl_100.txt \
#        -o $DATA/summarizer-output/exemplars/2012_Pakistan_garment_factory_fires/debug.max.lvecs100.txt \
#        --energy-log $DATA/summarizer-output/energy-logs/2012_Pakistan_garment_factory_fires.debug.max.lvecs100.tsv \
#        -r $DATA/summarizer-output/rouge/2012_Pakistan_garment_factory_fires.debug.max.lvecs100 \
#        -c $DATA/summarizer-output/clusters/2012_Pakistan_garment_factory_fires.debug.max.lvecs100.txt \
#        1>$TREC/logs/summarizer/2012_Pakistan_garment_factory_fires.debug.max.lvecs100.log \
#        2>$TREC/logs/summarizer/2012_Pakistan_garment_factory_fires.debug.max.lvecs100.err &
##else
##    echo "Skipping: Summarizing DEBUG NO-TEMP MAX ACCIDENTS 100d-lvecs" \
##    "2012_Pakistan_garment_factory_fires"
##fi
##
##wait
##
##if [ ! -f $DATA/summarizer-output/2012_Aurora_shooting/debug.max.lvecs100.txt ]; then
#    echo "Summarizing DEBUG NO-TEMP MAX TERRSHOOT 100d-lvecs" \
##    "2012_Aurora_shooting"
#    mkdir -p $TREC/logs/summarizer/
#    python -u $CUTTSUM/summarize-scripts/temporal_summarizer.py \
#        -b $DATA/summarizer-input-data/sentence-bow/2012_Aurora_shooting.tsv \
#        -l $DATA/salience-data/sentence-sims/2012_Aurora_shooting/terrshoot_spl_100.txt \
#        -o $DATA/summarizer-output/exemplars/2012_Aurora_shooting/debug.max.lvecs100.txt \
#        --energy-log $DATA/summarizer-output/energy-logs/2012_Aurora_shooting.debug.max.lvecs100.tsv \
#        -r $DATA/summarizer-output/rouge/2012_Aurora_shooting.debug.max.lvecs100 \
#        -c $DATA/summarizer-output/clusters/2012_Aurora_shooting.debug.max.lvecs100.txt \
#        1>$TREC/logs/summarizer/2012_Aurora_shooting.debug.max.lvecs100.log \
#        2>$TREC/logs/summarizer/2012_Aurora_shooting.debug.max.lvecs100.err &
##else
##    echo "Skipping: Summarizing DEBUG NO-TEMP MAX TERRSHOOT 100d-lvecs" \
##    "2012_Aurora_shooting"
##fi
##
##wait
##
##if [ ! -f $DATA/summarizer-output/Wisconsin_Sikh_temple_shooting/debug.max.lvecs100.txt ]; then
#    echo "Summarizing DEBUG NO-TEMP MAX TERRSHOOT 100d-lvecs" \
#    "Wisconsin_Sikh_temple_shooting"
#    mkdir -p $TREC/logs/summarizer/
#    python -u $CUTTSUM/summarize-scripts/temporal_summarizer.py \
#        -b $DATA/summarizer-input-data/sentence-bow/Wisconsin_Sikh_temple_shooting.tsv \
#        -l $DATA/salience-data/sentence-sims/Wisconsin_Sikh_temple_shooting/terrshoot_spl_100.txt \
#        -o $DATA/summarizer-output/exemplars/Wisconsin_Sikh_temple_shooting/debug.max.lvecs100.txt \
#        --energy-log $DATA/summarizer-output/energy-logs/Wisconsin_Sikh_temple_shooting.debug.max.lvecs100.tsv \
#        -r $DATA/summarizer-output/rouge/Wisconsin_Sikh_temple_shooting.debug.max.lvecs100 \
#        -c $DATA/summarizer-output/clusters/Wisconsin_Sikh_temple_shooting.debug.max.lvecs100.txt \
#        1>$TREC/logs/summarizer/Wisconsin_Sikh_temple_shooting.debug.max.lvecs100.log \
#        2>$TREC/logs/summarizer/Wisconsin_Sikh_temple_shooting.debug.max.lvecs100.err &
##else
##    echo "Skipping: Summarizing DEBUG NO-TEMP MAX TERRSHOOT 100d-lvecs" \
##    "Wisconsin_Sikh_temple_shooting"
##fi
##
##wait
##
##if [ ! -f $DATA/summarizer-output/Hurricane_Isaac_2012/debug.max.lvecs100.txt ]; then
#    echo "Summarizing DEBUG NO-TEMP MAX WEATHER 100d-lvecs" \
#    "Hurricane_Isaac_2012"
#    mkdir -p $TREC/logs/summarizer/
#    python -u $CUTTSUM/summarize-scripts/temporal_summarizer.py \
#        -b $DATA/summarizer-input-data/sentence-bow/Hurricane_Isaac_2012.tsv \
#        -l $DATA/salience-data/sentence-sims/Hurricane_Isaac_2012/weather_spl_100.txt \
#        -o $DATA/summarizer-output/exemplars/Hurricane_Isaac_2012/debug.max.lvecs100.txt \
#        --energy-log $DATA/summarizer-output/energy-logs/Hurricane_Isaac_2012.debug.max.lvecs100.tsv \
#        -r $DATA/summarizer-output/rouge/Hurricane_Isaac_2012.debug.max.lvecs100 \
#        -c $DATA/summarizer-output/clusters/Hurricane_Isaac_2012.debug.max.lvecs100.txt \
#        1>$TREC/logs/summarizer/Hurricane_Isaac_2012.debug.max.lvecs100.log \
#        2>$TREC/logs/summarizer/Hurricane_Isaac_2012.debug.max.lvecs100.err &
##else
##    echo "Skipping: Summarizing DEBUG NO-TEMP MAX WEATHER 100d-lvecs" \
##    "Hurricane_Isaac_2012"
##fi
##
##wait
##
##if [ ! -f $DATA/summarizer-output/Typhoon_Bopha/debug.max.lvecs100.txt ]; then
##    echo "Summarizing DEBUG NO-TEMP MAX WEATHER 100d-lvecs" \
##    "Typhoon_Bopha"
##    mkdir -p $TREC/logs/summarizer/
##    python -u $CUTTSUM/summarize-scripts/temporal_summarizer.py \
##        -b $DATA/summarizer-input-data/sentence-bow/Typhoon_Bopha.tsv \
##        -l $DATA/salience-data/sentence-sims/Typhoon_Bopha/weather_spl_100.txt \
##        -o $DATA/summarizer-output/Typhoon_Bopha/debug.max.lvecs100.txt \
##        --energy-log $DATA/summarizer-output/energy-logs/Typhoon_Bopha.debug.max.lvecs100.tsv \
##        1>$TREC/logs/summarizer/Typhoon_Bopha.debug.max.lvecs100.log \
##        2>$TREC/logs/summarizer/Typhoon_Bopha.debug.max.lvecs100.err &
##else
##    echo "Skipping: Summarizing DEBUG NO-TEMP MAX WEATHER 100d-lvecs" \
##    "Typhoon_Bopha"
##fi
##
##wait
##

#
#wait
#
#if [ ! -f $DATA/summarizer-output/2012_Tel_Aviv_bus_bombing/debug.max.lvecs100.txt ]; then
#    echo "Summarizing DEBUG NO-TEMP MAX TERRSHOOT 100d-lvecs" \
#    "2012_Tel_Aviv_bus_bombing"
#    mkdir -p $TREC/logs/summarizer/
#    python -u $CUTTSUM/summarize-scripts/temporal_summarizer.py \
#        -b $DATA/summarizer-input-data/sentence-bow/2012_Tel_Aviv_bus_bombing.tsv \
#        -l $DATA/salience-data/sentence-sims/2012_Tel_Aviv_bus_bombing/terrshoot_spl_100.txt \
#        -o $DATA/summarizer-output/2012_Tel_Aviv_bus_bombing/debug.max.lvecs100.txt \
#        --energy-log $DATA/summarizer-output/energy-logs/2012_Tel_Aviv_bus_bombing.debug.max.lvecs100.tsv \
#        1>$TREC/logs/summarizer/2012_Tel_Aviv_bus_bombing.debug.max.lvecs100.log \
#        2>$TREC/logs/summarizer/2012_Tel_Aviv_bus_bombing.debug.max.lvecs100.err &
#else
#    echo "Skipping: Summarizing DEBUG NO-TEMP MAX TERRSHOOT 100d-lvecs" \
#    "2012_Tel_Aviv_bus_bombing"
#fi
#
#
#wait
