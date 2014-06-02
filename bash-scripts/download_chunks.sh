DATA=/home/kedz/data/trec/
RES=$DATA/resources
python -u ../utility-scripts/download_chunks.py -c $RES/chunkurls.txt -e $RES/topics-masked.xml -w $DATA/event_chunks

