CHUNKS=$DATA/2013-event-chunks

if [ ! -d $TREC/logs/relevance1/ ]; then
    mkdir -p $TREC/logs/relevance1/
fi

if [ ! -d $DATA/relevant_chunks/filter1/2012_Buenos_Aires_Rail_Disaster ]; then
    echo "Extracting relevant (lvl 1) chunks for 2012_Buenos_Aires_Rail_Disaster..."
    python -u $CUTTSUM/install-scripts/python/extract_by_string_match.py \
        -e $DATA/events/test-events-2013.xml \
        -t "2012 Buenos Aires Rail Disaster" \
        -q "buenos aires( train| rail)? (crash|disaster|accident)" \
        -r $DATA/relevant_chunks/filter1/2012_Buenos_Aires_Rail_Disaster \
        -c $DATA/2013-event-chunks \
        -l $TREC/logs/relevance1/2012_Buenos_Aires_Rail_Disaster \
        --n-threads 4 \
        1>$TREC/logs/relevance1/2012_Buenos_Aires_Rail_Disaster.log \
        2>$TREC/logs/relevance1/2012_Buenos_Aires_Rail_Disaster.err &
else
    echo "Skipping: Extracting relevant (lvl 1) chunks for 2012_Buenos_Aires_Rail_Disaster..."
fi

if [ ! -d $DATA/relevant_chunks/filter1/2012_Pakistan_garment_factory_fires ]; then
    echo "Extracting relevant (lvl 1) chunks for 2012_Pakistan_garment_factory_fires..."
    python -u $CUTTSUM/install-scripts/python/extract_by_string_match.py \
        -e $DATA/events/test-events-2013.xml \
        -t "2012 Pakistan garment factory fires" \
        -q "pakistan( garment)? factory fires?" \
        -r $DATA/relevant_chunks/filter1/2012_Pakistan_garment_factory_fires \
        -c $DATA/2013-event-chunks \
        -l $TREC/logs/relevance1/2012_Pakistan_garment_factory_fires \
        --n-threads 4 \
        1>$TREC/logs/relevance1/2012_Pakistan_garment_factory_fires.log \
        2>$TREC/logs/relevance1/2012_Pakistan_garment_factory_fires.err &
else
    echo "Skipping: Extracting relevant (lvl 1) chunks for 2012_Pakistan_garment_factory_fires..."
fi

wait

if [ ! -d $DATA/relevant_chunks/filter1/2012_Aurora_shooting ]; then
    echo "Extracting relevant (lvl 1) chunks for 2012_Aurora_shooting..."
    python -u $CUTTSUM/install-scripts/python/extract_by_string_match.py \
        -e $DATA/events/test-events-2013.xml \
        -t "2012 Aurora shooting" \
        -q "colorado shooting" \
        -r $DATA/relevant_chunks/filter1/2012_Aurora_shooting \
        -c $DATA/2013-event-chunks \
        -l $TREC/logs/relevance1/2012_Aurora_shooting \
        --n-threads 2 \
        1>$TREC/logs/relevance1/2012_Aurora_shooting.log \
        2>$TREC/logs/relevance1/2012_Aurora_shooting.err &
else
    echo "Skipping: Extracting relevant (lvl 1) chunks for 2012_Aurora_shooting..."
fi

if [ ! -d $DATA/relevant_chunks/filter1/Wisconsin_Sikh_temple_shooting ]; then
    echo "Extracting relevant (lvl 1) chunks for Wisconsin_Sikh_temple_shooting..."
    mkdir -p $TREC/logs/relevance1/Wisconsin_Sikh_temple_shooting
    python -u $CUTTSUM/install-scripts/python/extract_by_string_match.py \
        -e $DATA/events/test-events-2013.xml \
        -t "Wisconsin Sikh temple shooting" \
        -q "(sikh )?temple shooting" \
        -r $DATA/relevant_chunks/filter1/Wisconsin_Sikh_temple_shooting \
        -c $DATA/2013-event-chunks \
        -l $TREC/logs/relevance1/Wisconsin_Sikh_temple_shooting \
        --n-threads 2 \
        1>$TREC/logs/relevance1/Wisconsin_Sikh_temple_shooting.log \
        2>$TREC/logs/relevance1/Wisconsin_Sikh_temple_shooting.err &
else
    echo "Skipping: Extracting relevant (lvl 1) chunks for Wisconsin_Sikh_temple_shooting..."
fi

if [ ! -d $DATA/relevant_chunks/filter1/Hurricane_Isaac_2012 ]; then
    echo "Extracting relevant (lvl 1) chunks for Hurricane_Isaac_2012..."
    mkdir -p $TREC/logs/relevance1/Hurricane_Isaac_2012
    python -u $CUTTSUM/install-scripts/python/extract_by_string_match.py \
        -e $DATA/events/test-events-2013.xml \
        -t "Hurricane Isaac (2012)" \
        -q "hurricane isaac" \
        -r $DATA/relevant_chunks/filter1/Hurricane_Isaac_2012 \
        -c $DATA/2013-event-chunks \
        -l $TREC/logs/relevance1/Hurricane_Isaac_2012 \
        --n-threads 2 \
        1>$TREC/logs/relevance1/Hurricane_Isaac_2012.log \
        2>$TREC/logs/relevance1/Hurricane_Isaac_2012.err &
else
    echo "Skipping: Extracting relevant (lvl 1) chunks for Hurricane_Isaac_2012..."
fi

if [ ! -d $DATA/relevant_chunks/filter1/Hurricane_Sandy ]; then
    echo "Extracting relevant (lvl 1) chunks for Hurricane_Sandy..."
    python -u $CUTTSUM/install-scripts/python/extract_by_string_match.py \
        -e $DATA/events/test-events-2013.xml \
        -t "Hurricane Sandy" \
        -q "hurricane sandy" \
        -r $DATA/relevant_chunks/filter1/Hurricane_Sandy \
        -c $DATA/2013-event-chunks \
        -l $TREC/logs/relevance1/Hurricane_Sandy \
        --n-threads 2 \
        1>$TREC/logs/relevance1/Hurricane_Sandy.log \
        2>$TREC/logs/relevance1/Hurricane_Sandy.err &
else
    echo "Skipping: Extracting relevant (lvl 1) chunks for Hurricane_Sandy..."
fi


if [ ! -d $DATA/relevant_chunks/filter1/Typhoon_Bopha ]; then
    echo "Extracting relevant (lvl 1) chunks for Typhoon_Bopha..."
    python -u $CUTTSUM/install-scripts/python/extract_by_string_match.py \
        -e $DATA/events/test-events-2013.xml \
        -t "Typhoon Bopha" \
        -q "typhoon bopha" \
        -r $DATA/relevant_chunks/filter1/Typhoon_Bopha \
        -c $DATA/2013-event-chunks \
        -l $TREC/logs/relevance1/Typhoon_Bopha \
        --n-threads 2 \
        1>$TREC/logs/relevance1/Typhoon_Bopha.log \
        2>$TREC/logs/relevance1/Typhoon_Bopha.err &
else
    echo "Skipping: Extracting relevant (lvl 1) chunks for Typhoon_Bopha..."
fi

if [ ! -d $DATA/relevant_chunks/filter1/2012_Guatemala_earthquake ]; then
    echo "Extracting relevant (lvl 1) chunks for 2012_Guatemala_earthquake..."
    python -u $CUTTSUM/install-scripts/python/extract_by_string_match.py \
        -e $DATA/events/test-events-2013.xml \
        -t "2012 Guatemala earthquake" \
        -q "guatemala earthquake" \
        -r $DATA/relevant_chunks/filter1/2012_Guatemala_earthquake \
        -c $DATA/2013-event-chunks \
        -l $TREC/logs/relevance1/2012_Guatemala_earthquake \
        --n-threads 2 \
        1>$TREC/logs/relevance1/2012_Guatemala_earthquake.log \
        2>$TREC/logs/relevance1/2012_Guatemala_earthquake.err &
else
    echo "Skipping: Extracting relevant (lvl 1) chunks for 2012_Guatemala_earthquake..."
fi

if [ ! -d $DATA/relevant_chunks/filter1/2012_Tel_Aviv_bus_bombing ]; then
    echo "Extracting relevant (lvl 1) chunks for 2012_Tel_Aviv_bus_bombing..."
    python -u $CUTTSUM/install-scripts/python/extract_by_string_match.py \
        -e $DATA/events/test-events-2013.xml \
        -t "2012 Tel Aviv bus bombing" \
        -q "tel aviv( bus)? bombing" \
        -r $DATA/relevant_chunks/filter1/2012_Tel_Aviv_bus_bombing \
        -c $DATA/2013-event-chunks \
        -l $TREC/logs/relevance1/2012_Tel_Aviv_bus_bombing \
        --n-threads 2 \
        1>$TREC/logs/relevance1/2012_Tel_Aviv_bus_bombing.log \
        2>$TREC/logs/relevance1/2012_Tel_Aviv_bus_bombing.err &
else
    echo "Skipping: Extracting relevant (lvl 1) chunks for 2012_Tel_Aviv_bus_bombing..."
fi

wait




if [ ! -d $DATA/relevant_chunks/filter2/2012_Buenos_Aires_Rail_Disaster ]; then
    echo "Extracting content (lvl 2) for 2012_Buenos_Aires_Rail_Disaster"
    mkdir -p $TREC/logs/relevance2/2012_Buenos_Aires_Rail_Disaster
    python -u $CUTTSUM/install-scripts/python/extract_content.py \
        -e $DATA/events/test-events-2013.xml \
        -t "2012 Buenos Aires Rail Disaster"  \
        -r $DATA/relevant_chunks/filter1/2012_Buenos_Aires_Rail_Disaster \
        -o $DATA/relevant_chunks/filter2/2012_Buenos_Aires_Rail_Disaster \
        -a $DATA/models/article-detector \
        -j 3 \
        -l $TREC/logs/relevance2/2012_Buenos_Aires_Rail_Disaster \
        1>$TREC/logs/relevance2/2012_Buenos_Aires_Rail_Disaster.log \
        2>$TREC/logs/relevance2/2012_Buenos_Aires_Rail_Disaster.err &

else
    echo "Skipping: Extracting article content (lvl 2) for 2012_Buenos_Aires_Rail_Disaster"
fi

if [ ! -d $DATA/relevant_chunks/filter2/2012_Pakistan_garment_factory_fires ]; then
    echo "Extracting content (lvl 2) for 2012_Pakistan_garment_factory_fires"
    mkdir -p $TREC/logs/relevance2/2012_Pakistan_garment_factory_fires
    python -u $CUTTSUM/install-scripts/python/extract_content.py \
        -e $DATA/events/test-events-2013.xml \
        -t "2012 Pakistan garment factory fires"  \
        -r $DATA/relevant_chunks/filter1/2012_Pakistan_garment_factory_fires \
        -o $DATA/relevant_chunks/filter2/2012_Pakistan_garment_factory_fires \
        -a $DATA/models/article-detector \
        -j 3 \
        -l $TREC/logs/relevance2/2012_Pakistan_garment_factory_fires \
        1>$TREC/logs/relevance2/2012_Pakistan_garment_factory_fires.log \
        2>$TREC/logs/relevance2/2012_Pakistan_garment_factory_fires.err &

else
    echo "Skipping: Extracting article content (lvl 2) for 2012_Pakistan_garment_factory_fires"
fi

if [ ! -d $DATA/relevant_chunks/filter2/2012_Aurora_shooting ]; then
    echo "Extracting content (lvl 2) for 2012_Aurora_shooting"
    mkdir -p $TREC/logs/relevance2/2012_Aurora_shooting
    python -u $CUTTSUM/install-scripts/python/extract_content.py \
        -e $DATA/events/test-events-2013.xml \
        -t "2012 Aurora shooting"  \
        -r $DATA/relevant_chunks/filter1/2012_Aurora_shooting \
        -o $DATA/relevant_chunks/filter2/2012_Aurora_shooting \
        -a $DATA/models/article-detector \
        -j 3 \
        -l $TREC/logs/relevance2/2012_Aurora_shooting \
        1>$TREC/logs/relevance2/2012_Aurora_shooting.log \
        2>$TREC/logs/relevance2/2012_Aurora_shooting.err &

else
    echo "Skipping: Extracting article content (lvl 2) for 2012_Aurora_shooting"
fi

if [ ! -d $DATA/relevant_chunks/filter2/Wisconsin_Sikh_temple_shooting ]; then
    echo "Extracting content (lvl 2) for Wisconsin_Sikh_temple_shooting"
    mkdir -p $TREC/logs/relevance2/Wisconsin_Sikh_temple_shooting
    python -u $CUTTSUM/install-scripts/python/extract_content.py \
        -e $DATA/events/test-events-2013.xml \
        -t "Wisconsin Sikh temple shooting"  \
        -r $DATA/relevant_chunks/filter1/Wisconsin_Sikh_temple_shooting \
        -o $DATA/relevant_chunks/filter2/Wisconsin_Sikh_temple_shooting \
        -a $DATA/models/article-detector \
        -j 3 \
        -l $TREC/logs/relevance2/Wisconsin_Sikh_temple_shooting \
        1>$TREC/logs/relevance2/Wisconsin_Sikh_temple_shooting.log \
        2>$TREC/logs/relevance2/Wisconsin_Sikh_temple_shooting.err &

else
    echo "Skipping: Extracting article content (lvl 2) for Wisconsin_Sikh_temple_shooting"
fi

if [ ! -d $DATA/relevant_chunks/filter2/Hurricane_Isaac_2012 ]; then
    echo "Extracting content (lvl 2) for Hurricane_Isaac_2012"
    mkdir -p $TREC/logs/relevance2/Hurricane_Isaac_2012
    python -u $CUTTSUM/install-scripts/python/extract_content.py \
        -e $DATA/events/test-events-2013.xml \
        -t "Hurricane Isaac (2012)"  \
        -r $DATA/relevant_chunks/filter1/Hurricane_Isaac_2012 \
        -o $DATA/relevant_chunks/filter2/Hurricane_Isaac_2012 \
        -a $DATA/models/article-detector \
        -j 3 \
        -l $TREC/logs/relevance2/Hurricane_Isaac_2012 \
        1>$TREC/logs/relevance2/Hurricane_Isaac_2012.log \
        2>$TREC/logs/relevance2/Hurricane_Isaac_2012.err &

else
    echo "Skipping: Extracting article content (lvl 2) for Hurricane_Isaac_2012"
fi

if [ ! -d $DATA/relevant_chunks/filter2/Hurricane_Sandy ]; then
    echo "Extracting content (lvl 2) for Hurricane_Sandy"
    mkdir -p $TREC/logs/relevance2/Hurricane_Sandy
    python -u $CUTTSUM/install-scripts/python/extract_content.py \
        -e $DATA/events/test-events-2013.xml \
        -t "Hurricane Sandy"  \
        -r $DATA/relevant_chunks/filter1/Hurricane_Sandy \
        -o $DATA/relevant_chunks/filter2/Hurricane_Sandy \
        -a $DATA/models/article-detector \
        -j 3 \
        -l $TREC/logs/relevance2/Hurricane_Sandy \
        1>$TREC/logs/relevance2/Hurricane_Sandy.log \
        2>$TREC/logs/relevance2/Hurricane_Sandy.err &

else
    echo "Skipping: Extracting article content (lvl 2) for Hurricane_Sandy"
fi


if [ ! -d $DATA/relevant_chunks/filter2/Typhoon_Bopha ]; then
    echo "Extracting content (lvl 2) for Typhoon_Bopha"
    mkdir -p $TREC/logs/relevance2/Typhoon_Bopha
    python -u $CUTTSUM/install-scripts/python/extract_content.py \
        -e $DATA/events/test-events-2013.xml \
        -t "Typhoon Bopha"  \
        -r $DATA/relevant_chunks/filter1/Typhoon_Bopha \
        -o $DATA/relevant_chunks/filter2/Typhoon_Bopha \
        -a $DATA/models/article-detector \
        -j 3 \
        -l $TREC/logs/relevance2/Typhoon_Bopha \
        1>$TREC/logs/relevance2/Typhoon_Bopha.log \
        2>$TREC/logs/relevance2/Typhoon_Bopha.err &

else
    echo "Skipping: Extracting article content (lvl 2) for Typhoon_Bopha"
fi

if [ ! -d $DATA/relevant_chunks/filter2/2012_Guatemala_earthquake ]; then
    echo "Extracting content (lvl 2) for 2012_Guatemala_earthquake"
    mkdir -p $TREC/logs/relevance2/2012_Guatemala_earthquake
    python -u $CUTTSUM/install-scripts/python/extract_content.py \
        -e $DATA/events/test-events-2013.xml \
        -t "2012 Guatemala earthquake"  \
        -r $DATA/relevant_chunks/filter1/2012_Guatemala_earthquake \
        -o $DATA/relevant_chunks/filter2/2012_Guatemala_earthquake \
        -a $DATA/models/article-detector \
        -j 3 \
        -l $TREC/logs/relevance2/2012_Guatemala_earthquake \
        1>$TREC/logs/relevance2/2012_Guatemala_earthquake.log \
        2>$TREC/logs/relevance2/2012_Guatemala_earthquake.err &

else
    echo "Skipping: Extracting article content (lvl 2) for 2012_Guatemala_earthquake"
fi

if [ ! -d $DATA/relevant_chunks/filter2/2012_Tel_Aviv_bus_bombing ]; then
    echo "Extracting content (lvl 2) for 2012_Tel_Aviv_bus_bombing"
    mkdir -p $TREC/logs/relevance2/2012_Tel_Aviv_bus_bombing
    python -u $CUTTSUM/install-scripts/python/extract_content.py \
        -e $DATA/events/test-events-2013.xml \
        -t "2012 Tel Aviv bus bombing"  \
        -r $DATA/relevant_chunks/filter1/2012_Tel_Aviv_bus_bombing \
        -o $DATA/relevant_chunks/filter2/2012_Tel_Aviv_bus_bombing \
        -a $DATA/models/article-detector \
        -j 3 \
        -l $TREC/logs/relevance2/2012_Tel_Aviv_bus_bombing \
        1>$TREC/logs/relevance2/2012_Tel_Aviv_bus_bombing.log \
        2>$TREC/logs/relevance2/2012_Tel_Aviv_bus_bombing.err &

else
    echo "Skipping: Extracting article content (lvl 2) for 2012_Tel_Aviv_bus_bombing"
fi

wait
