if [ ! -f $DATA/salience-data/sentence-features/2012_Buenos_Aires_Rail_Disaster.txt ]; then
    echo "Computing sentence features for 2012_Buenos_Aires_Rail_Disaster..."
    mkdir -p $TREC/logs/compute-sent-feats/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_features.py \
        -t "2012 Buenos Aires Rail Disaster" \
        -r $DATA/relevant_chunks/filter2/2012_Buenos_Aires_Rail_Disaster \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/salience-data/sentence-features/2012_Buenos_Aires_Rail_Disaster.txt \
        --domain-lm-port 9903 \
        --background-lm-ports 9905 9906 9907 \
        --word-freqs $DATA/word-counts \
        --doc-freqs $DATA/doc-counts \
        1>$TREC/logs/compute-sent-feats/2012_Buenos_Aires_Rail_Disaster.log \
        2>$TREC/logs/compute-sent-feats/2012_Buenos_Aires_Rail_Disaster.err &
else
    echo "Skipping: Computing sentence features for 2012_Buenos_Aires_Rail_Disaster..."
fi

if [ ! -f $DATA/salience-data/sentence-features/2012_Pakistan_garment_factory_fires.txt ]; then
    echo "Computing sentence features for 2012_Pakistan_garment_factory_fires..."
    mkdir -p $TREC/logs/compute-sent-feats/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_features.py \
        -t "2012 Pakistan garment factory fires" \
        -r $DATA/relevant_chunks/filter2/2012_Pakistan_garment_factory_fires \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/salience-data/sentence-features/2012_Pakistan_garment_factory_fires.txt \
        --domain-lm-port 9903 \
        --background-lm-ports 9905 9906 9907 \
        --word-freqs $DATA/word-counts \
        --doc-freqs $DATA/doc-counts \
        1>$TREC/logs/compute-sent-feats/2012_Pakistan_garment_factory_fires.log \
        2>$TREC/logs/compute-sent-feats/2012_Pakistan_garment_factory_fires.err &
else
    echo "Skipping: Computing sentence features for 2012_Pakistan_garment_factory_fires..."
fi

if [ ! -f $DATA/salience-data/sentence-features/2012_Aurora_shooting.txt ]; then
    echo "Computing sentence features for 2012_Aurora_shooting..."
    mkdir -p $TREC/logs/compute-sent-feats/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_features.py \
        -t "2012 Aurora shooting" \
        -r $DATA/relevant_chunks/filter2/2012_Aurora_shooting \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/salience-data/sentence-features/2012_Aurora_shooting.txt \
        --domain-lm-port 9902 \
        --background-lm-ports 9905 9906 9907 \
        --word-freqs $DATA/word-counts \
        --doc-freqs $DATA/doc-counts \
        1>$TREC/logs/compute-sent-feats/2012_Aurora_shooting.log \
        2>$TREC/logs/compute-sent-feats/2012_Aurora_shooting.err &
else
    echo "Skipping: Computing sentence features for 2012_Aurora_shooting..."
fi

if [ ! -f $DATA/salience-data/sentence-features/Wisconsin_Sikh_temple_shooting.txt ]; then
    echo "Computing sentence features for Wisconsin_Sikh_temple_shooting..."
    mkdir -p $TREC/logs/compute-sent-feats/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_features.py \
        -t "Wisconsin Sikh temple shooting" \
        -r $DATA/relevant_chunks/filter2/Wisconsin_Sikh_temple_shooting \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/salience-data/sentence-features/Wisconsin_Sikh_temple_shooting.txt \
        --domain-lm-port 9902 \
        --background-lm-ports 9905 9906 9907 \
        --word-freqs $DATA/word-counts \
        --doc-freqs $DATA/doc-counts \
        1>$TREC/logs/compute-sent-feats/Wisconsin_Sikh_temple_shooting.log \
        2>$TREC/logs/compute-sent-feats/Wisconsin_Sikh_temple_shooting.err &
else
    echo "Skipping: Computing sentence features for Wisconsin_Sikh_temple_shooting..."
fi

if [ ! -f $DATA/salience-data/sentence-features/Hurricane_Isaac_2012.txt ]; then
    echo "Computing sentence features for Hurricane_Isaac_2012..."
    mkdir -p $TREC/logs/compute-sent-feats/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_features.py \
        -t "Hurricane Isaac (2012)" \
        -r $DATA/relevant_chunks/filter2/Hurricane_Isaac_2012 \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/salience-data/sentence-features/Hurricane_Isaac_2012.txt \
        --domain-lm-port 9901 \
        --background-lm-ports 9905 9906 9907 \
        --word-freqs $DATA/word-counts \
        --doc-freqs $DATA/doc-counts \
        1>$TREC/logs/compute-sent-feats/Hurricane_Isaac_2012.log \
        2>$TREC/logs/compute-sent-feats/Hurricane_Isaac_2012.err &
else
    echo "Skipping: Computing sentence features for Hurricane_Isaac_2012..."
fi

if [ ! -f $DATA/salience-data/sentence-features/Hurricane_Sandy.txt ]; then
    echo "Computing sentence features for Hurricane_Sandy..."
    mkdir -p $TREC/logs/compute-sent-feats/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_features.py \
        -t "Hurricane Sandy" \
        -r $DATA/relevant_chunks/filter2/Hurricane_Sandy \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/salience-data/sentence-features/Hurricane_Sandy.txt \
        --domain-lm-port 9901 \
        --background-lm-ports 9905 9906 9907 \
        --word-freqs $DATA/word-counts \
        --doc-freqs $DATA/doc-counts \
        1>$TREC/logs/compute-sent-feats/Hurricane_Sandy.log \
        2>$TREC/logs/compute-sent-feats/Hurricane_Sandy.err &
else
    echo "Skipping: Computing sentence features for Hurricane_Sandy..."
fi

if [ ! -f $DATA/salience-data/sentence-features/June_2012_North_American_derecho.txt ]; then
    echo "Computing sentence features for June_2012_North_American_derecho..."
    mkdir -p $TREC/logs/compute-sent-feats/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_features.py \
        -t "June 2012 North American derecho" \
        -r $DATA/relevant_chunks/filter2/June_2012_North_American_derecho \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/salience-data/sentence-features/June_2012_North_American_derecho.txt \
        --domain-lm-port 9901 \
        --background-lm-ports 9905 9906 9907 \
        --word-freqs $DATA/word-counts \
        --doc-freqs $DATA/doc-counts \
        1>$TREC/logs/compute-sent-feats/June_2012_North_American_derecho.log \
        2>$TREC/logs/compute-sent-feats/June_2012_North_American_derecho.err &
else
    echo "Skipping: Computing sentence features for June_2012_North_American_derecho..."
fi

if [ ! -f $DATA/salience-data/sentence-features/Typhoon_Bopha.txt ]; then
    echo "Computing sentence features for Typhoon_Bopha..."
    mkdir -p $TREC/logs/compute-sent-feats/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_features.py \
        -t "Typhoon Bopha" \
        -r $DATA/relevant_chunks/filter2/Typhoon_Bopha \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/salience-data/sentence-features/Typhoon_Bopha.txt \
        --domain-lm-port 9901 \
        --background-lm-ports 9905 9906 9907 \
        --word-freqs $DATA/word-counts \
        --doc-freqs $DATA/doc-counts \
        1>$TREC/logs/compute-sent-feats/Typhoon_Bopha.log \
        2>$TREC/logs/compute-sent-feats/Typhoon_Bopha.err &
else
    echo "Skipping: Computing sentence features for Typhoon_Bopha..."
fi

if [ ! -f $DATA/salience-data/sentence-features/2012_Guatemala_earthquake.txt ]; then
    echo "Computing sentence features for 2012_Guatemala_earthquake..."
    mkdir -p $TREC/logs/compute-sent-feats/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_features.py \
        -t "2012 Guatemala earthquake" \
        -r $DATA/relevant_chunks/filter2/2012_Guatemala_earthquake \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/salience-data/sentence-features/2012_Guatemala_earthquake.txt \
        --domain-lm-port 9900 \
        --background-lm-ports 9905 9906 9907 \
        --word-freqs $DATA/word-counts \
        --doc-freqs $DATA/doc-counts \
        1>$TREC/logs/compute-sent-feats/2012_Guatemala_earthquake.log \
        2>$TREC/logs/compute-sent-feats/2012_Guatemala_earthquake.err &
else
    echo "Skipping: Computing sentence features for 2012_Guatemala_earthquake..."
fi

if [ ! -f $DATA/salience-data/sentence-features/2012_Tel_Aviv_bus_bombing.txt ]; then
    echo "Computing sentence features for 2012_Tel_Aviv_bus_bombing..."
    mkdir -p $TREC/logs/compute-sent-feats/
    python -u $CUTTSUM/model-training-scripts/python/compute_sentence_features.py \
        -t "2012 Tel Aviv bus bombing" \
        -r $DATA/relevant_chunks/filter2/2012_Tel_Aviv_bus_bombing \
        -e $DATA/events/test-events-2013.xml \
        -o $DATA/salience-data/sentence-features/2012_Tel_Aviv_bus_bombing.txt \
        --domain-lm-port 9902 \
        --background-lm-ports 9905 9906 9907 \
        --word-freqs $DATA/word-counts \
        --doc-freqs $DATA/doc-counts \
        1>$TREC/logs/compute-sent-feats/2012_Tel_Aviv_bus_bombing.log \
        2>$TREC/logs/compute-sent-feats/2012_Tel_Aviv_bus_bombing.err &
else
    echo "Skipping: Computing sentence features for 2012_Tel_Aviv_bus_bombing..."
fi

wait
