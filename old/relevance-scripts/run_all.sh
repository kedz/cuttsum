source $TREC_VENV/bin/activate

#python -u $CUTTSUM/relevance-scripts/extract_by_string_match.py -e $TREC/data/2013-events/topics-masked.xml -t "2012 Guatemala earthquake" -q "Guatemalan? earthquake" -r $TREC/data/relevant_chunks/2012_Guatemala_earthquake -c $TREC/data/chunks -l $TREC/logs --n-threads 24

python -u $CUTTSUM/relevance-scripts/extract_relevance_regression_features.py -e $TREC/data/2013-events/topics-masked.xml -t "2012 Guatemala earthquake" -r $TREC/data/relevant_chunks/2012_Guatemala_earthquake  -n $TREC/data/2013-data/nuggets.tsv  -d $TREC/data/doc-frequencies -w $TREC/data/word_frequencies  #-l $TREC/logs --n-threads 24
