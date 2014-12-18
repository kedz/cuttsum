if [ ! -f $LM_INPUT/gigaword_lm_input.txt ]; then
    echo "Extracting Gigaword text..."
    python -u $CUTTSUM/install-scripts/python/gigaword2txt.py -g $GIGAWORD_DATA \
        -of $LM_INPUT/gigaword_lm_input.txt \
        -r '(nyt|apw)_eng_(2009|2008|2007|2006|2005|2004|2003|2002|2001|2000|1999)' \
        1>$TREC/logs/gw_lm_input.log 2>$TREC/logs/gw_lm_input.err
else
    echo "Skipping: Extracting Gigaword text..."
fi

if [ ! -f $LM_INPUT/gigaword_vocab.txt ]; then
    echo "Generating vocab file"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/gigaword_lm_input.txt \
        -of $LM_INPUT/gigaword_vocab.txt -t 3 
else
    echo "Skipping: Generating vocab file"
fi

if [ ! -d $LM/gigaword ]; then
    mkdir -p $LM/gigaword
fi

if [ ! -f $LM/gigaword/gigaword_3.arpa ]; then
    echo "Training gigaword 3-gram lm model..."
    ngram-count -order 3 -kndiscount -interpolate \
        -text $LM_INPUT/gigaword_lm_input.txt \
        -vocab $LM_INPUT/gigaword_vocab.txt \
        -lm $LM/gigaword/gigaword_3.arpa &
else
    echo "Skipping: Training gigaword 3-gram lm model..."
fi

if [ ! -f $LM/gigaword/gigaword_4.arpa ]; then
    echo "Training gigaword 4-gram lm model..."
    ngram-count -order 4 -kndiscount -interpolate \
        -text $LM_INPUT/gigaword_lm_input.txt \
        -vocab $LM_INPUT/gigaword_vocab.txt \
        -lm $LM/gigaword/gigaword_4.arpa &
else
    echo "Skipping: Training gigaword 4-gram lm model..."
fi

if [ ! -f $LM/gigaword/gigaword_5.arpa ]; then
    echo "Training gigaword 5-gram lm model..."
    ngram-count -order 5 -kndiscount -interpolate \
        -text $LM_INPUT/gigaword_lm_input.txt \
        -vocab $LM_INPUT/gigaword_vocab.txt \
        -lm $LM/gigaword/gigaword_5.arpa &
else
    echo "Skipping: Training gigaword 5-gram lm model..."
fi

wait
