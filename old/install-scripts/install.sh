source $TREC_VENV/bin/activate
export DATA=$TREC/data2
export LM_INPUT=$DATA/lm_input
export LM=$DATA/lm
export LM_INPUT_LOG=$TREC/logs/lm_input.log
export WGET_LOG=$TREC/logs/wgets.log

if [ ! -d $TREC/logs ]; then
    mkdir -p $TREC/logs
fi

if [ ! -d $DATA ]; then
    echo "Making data directory: $DATA"
    mkdir -p $DATA
fi

echo "INSTALLING TREC RESOURCES"
echo "========================="
bash $CUTTSUM/install-scripts/bash/download-trec-resources.sh
echo

echo "INSTALLING GIGAWORD LANGUAGE MODELS"
echo "==================================="
bash $CUTTSUM/install-scripts/bash/install-gigaword-lm.sh
echo

echo "DOWNLOADING WIKIPEDIA PAGE LISTS"
echo "================================"
echo "NOT IMPLEMENTED"
echo

echo "DOWNLOAD WIKIPEDIA CATEGORY CORPORA"
echo "==================================="
bash $CUTTSUM/install-scripts/bash/download-wiki-html.sh
echo

echo "EXTRACTING TXT FROM WP HTML FILES"
echo "================================="
bash $CUTTSUM/install-scripts/bash/extract-wp-txt-from-html.sh
echo

echo "INSTALL WIKIPEDIA LANGUAGE MODELS"
echo "================================="
bash $CUTTSUM/install-scripts/bash/install-wikipedia-lm.sh
echo

echo "GENERATE BACKGROUND WORD/DOC COUNTS"
echo "==================================="
if [ ! -d $DATA/word-counts ]; then 
    echo "Computing background counts..."
    python -u $CUTTSUM/install-scripts/python/compute_background_word_counts.py \
        -c $DATA/2013-event-chunks \
        -w $DATA/word-counts \
        -d $DATA/doc-counts \
        -n 20 \
        -l $TREC/logs/word_freqs 
else
    echo "Skipping: Computing background counts..."
fi
echo

echo "TRAINING SENTENCE SIMILARITY MODELS"
echo "==================================="
bash $CUTTSUM/install-scripts/bash/train-sentence-sim-models.sh
echo

echo "EXTRACT RELEVANT CHUNKS"
echo "======================="
bash $CUTTSUM/install-scripts/bash/extract-relevant-chunks.sh
echo


#echo "STARTING LM SERVERS"
#echo "==================="
#bash $CUTTSUM/install-scripts/bash/start-lm-servers.sh
#echo


echo "Finished loading data and building models."
