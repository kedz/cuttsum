if [ ! -f $DATA/summarizer-input-data/sentence-bow/2012_Buenos_Aires_Rail_Disaster.tsv ]; then
    echo "Computing sentence bow for 2012_Buenos_Aires_Rail_Disaster..."
    mkdir -p $TREC/logs/compute-sent-bow/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_bow.py \
        -t "2012 Buenos Aires Rail Disaster" \
        -r $DATA/relevant_chunks/filter2/2012_Buenos_Aires_Rail_Disaster \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/summarizer-input-data/sentence-bow/2012_Buenos_Aires_Rail_Disaster.tsv \
        1>$TREC/logs/compute-sent-bow/2012_Buenos_Aires_Rail_Disaster.log \
        2>$TREC/logs/compute-sent-bow/2012_Buenos_Aires_Rail_Disaster.err &
else
    echo "Skipping: Computing sentence bow for 2012_Buenos_Aires_Rail_Disaster..."
fi

if [ ! -f $DATA/summarizer-input-data/sentence-bow/2012_Pakistan_garment_factory_fires.tsv ]; then
    echo "Computing sentence bow for 2012_Pakistan_garment_factory_fires..."
    mkdir -p $TREC/logs/compute-sent-bow/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_bow.py \
        -t "2012 Pakistan garment factory fires" \
        -r $DATA/relevant_chunks/filter2/2012_Pakistan_garment_factory_fires \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/summarizer-input-data/sentence-bow/2012_Pakistan_garment_factory_fires.tsv \
        1>$TREC/logs/compute-sent-bow/2012_Pakistan_garment_factory_fires.log \
        2>$TREC/logs/compute-sent-bow/2012_Pakistan_garment_factory_fires.err &
else
    echo "Skipping: Computing sentence bow for 2012_Pakistan_garment_factory_fires..."
fi

if [ ! -f $DATA/summarizer-input-data/sentence-bow/2012_Aurora_shooting.tsv ]; then
    echo "Computing sentence bow for 2012_Aurora_shooting..."
    mkdir -p $TREC/logs/compute-sent-bow/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_bow.py \
        -t "2012 Aurora shooting" \
        -r $DATA/relevant_chunks/filter2/2012_Aurora_shooting \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/summarizer-input-data/sentence-bow/2012_Aurora_shooting.tsv \
        1>$TREC/logs/compute-sent-bow/2012_Aurora_shooting.log \
        2>$TREC/logs/compute-sent-bow/2012_Aurora_shooting.err &
else
    echo "Skipping: Computing sentence bow for 2012_Aurora_shooting..."
fi

if [ ! -f $DATA/summarizer-input-data/sentence-bow/Wisconsin_Sikh_temple_shooting.tsv ]; then
    echo "Computing sentence bow for Wisconsin_Sikh_temple_shooting..."
    mkdir -p $TREC/logs/compute-sent-bow/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_bow.py \
        -t "Wisconsin Sikh temple shooting" \
        -r $DATA/relevant_chunks/filter2/Wisconsin_Sikh_temple_shooting \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/summarizer-input-data/sentence-bow/Wisconsin_Sikh_temple_shooting.tsv \
        1>$TREC/logs/compute-sent-bow/Wisconsin_Sikh_temple_shooting.log \
        2>$TREC/logs/compute-sent-bow/Wisconsin_Sikh_temple_shooting.err &
else
    echo "Skipping: Computing sentence bow for Wisconsin_Sikh_temple_shooting..."
fi

if [ ! -f $DATA/summarizer-input-data/sentence-bow/Hurricane_Isaac_2012.tsv ]; then
    echo "Computing sentence bow for Hurricane_Isaac_2012..."
    mkdir -p $TREC/logs/compute-sent-bow/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_bow.py \
        -t "Hurricane Isaac (2012)" \
        -r $DATA/relevant_chunks/filter2/Hurricane_Isaac_2012 \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/summarizer-input-data/sentence-bow/Hurricane_Isaac_2012.tsv \
        1>$TREC/logs/compute-sent-bow/Hurricane_Isaac_2012.log \
        2>$TREC/logs/compute-sent-bow/Hurricane_Isaac_2012.err &
else
    echo "Skipping: Computing sentence bow for Hurricane_Isaac_2012..."
fi

if [ ! -f $DATA/summarizer-input-data/sentence-bow/Hurricane_Sandy.tsv ]; then
    echo "Computing sentence bow for Hurricane_Sandy..."
    mkdir -p $TREC/logs/compute-sent-bow/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_bow.py \
        -t "Hurricane Sandy" \
        -r $DATA/relevant_chunks/filter2/Hurricane_Sandy \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/summarizer-input-data/sentence-bow/Hurricane_Sandy.tsv \
        1>$TREC/logs/compute-sent-bow/Hurricane_Sandy.log \
        2>$TREC/logs/compute-sent-bow/Hurricane_Sandy.err &
else
    echo "Skipping: Computing sentence bow for Hurricane_Sandy..."
fi

if [ ! -f $DATA/summarizer-input-data/sentence-bow/June_2012_North_American_derecho.tsv ]; then
    echo "Computing sentence bow for June_2012_North_American_derecho..."
    mkdir -p $TREC/logs/compute-sent-bow/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_bow.py \
        -t "June 2012 North American derecho" \
        -r $DATA/relevant_chunks/filter2/June_2012_North_American_derecho \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/summarizer-input-data/sentence-bow/June_2012_North_American_derecho.tsv \
        1>$TREC/logs/compute-sent-bow/June_2012_North_American_derecho.log \
        2>$TREC/logs/compute-sent-bow/June_2012_North_American_derecho.err &
else
    echo "Skipping: Computing sentence bow for June_2012_North_American_derecho..."
fi

if [ ! -f $DATA/summarizer-input-data/sentence-bow/Typhoon_Bopha.tsv ]; then
    echo "Computing sentence bow for Typhoon_Bopha..."
    mkdir -p $TREC/logs/compute-sent-bow/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_bow.py \
        -t "Typhoon Bopha" \
        -r $DATA/relevant_chunks/filter2/Typhoon_Bopha \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/summarizer-input-data/sentence-bow/Typhoon_Bopha.tsv \
        1>$TREC/logs/compute-sent-bow/Typhoon_Bopha.log \
        2>$TREC/logs/compute-sent-bow/Typhoon_Bopha.err &
else
    echo "Skipping: Computing sentence bow for Typhoon_Bopha..."
fi

if [ ! -f $DATA/summarizer-input-data/sentence-bow/2012_Guatemala_earthquake.tsv ]; then
    echo "Computing sentence bow for 2012_Guatemala_earthquake..."
    mkdir -p $TREC/logs/compute-sent-bow/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_bow.py \
        -t "2012 Guatemala earthquake" \
        -r $DATA/relevant_chunks/filter2/2012_Guatemala_earthquake \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/summarizer-input-data/sentence-bow/2012_Guatemala_earthquake.tsv \
        1>$TREC/logs/compute-sent-bow/2012_Guatemala_earthquake.log \
        2>$TREC/logs/compute-sent-bow/2012_Guatemala_earthquake.err &
else
    echo "Skipping: Computing sentence bow for 2012_Guatemala_earthquake..."
fi

if [ ! -f $DATA/summarizer-input-data/sentence-bow/2012_Tel_Aviv_bus_bombing.tsv ]; then
    echo "Computing sentence bow for 2012_Tel_Aviv_bus_bombing..."
    mkdir -p $TREC/logs/compute-sent-bow/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_bow.py \
        -t "2012 Tel Aviv bus bombing" \
        -r $DATA/relevant_chunks/filter2/2012_Tel_Aviv_bus_bombing \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/summarizer-input-data/sentence-bow/2012_Tel_Aviv_bus_bombing.tsv \
        1>$TREC/logs/compute-sent-bow/2012_Tel_Aviv_bus_bombing.log \
        2>$TREC/logs/compute-sent-bow/2012_Tel_Aviv_bus_bombing.err &
else
    echo "Skipping: Computing sentence bow for 2012_Tel_Aviv_bus_bombing..."
fi

wait
