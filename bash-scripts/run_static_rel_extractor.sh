DATA=/home/kedz/data/trec
RES=$DATA/resources
OUT=/home/kedz/results/trec/relevant
STATS=/home/kedz/results/trec/stats

python -u python/static_relevance_pipeline.py -e $RES/training-topics.xml -c $DATA/training_chunks -t "2012 East Azerbaijan earthquakes" -o $OUT/2012_East_Azerbaijan_earthquakes.sc.xz -s $STATS/2012_East_Azerbaijan_earthquakes --n-threads 4
