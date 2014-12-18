ngram -lm $DATA/lm/domain/earthquakes_3.arpa -unk \
    -vocab $DATA/lm_input/earthquakes_wiki_simple_vocab.txt \
    -server-port 9900 &
PID=$!
DIR=`pwd`
echo "kill $PID" > $CUTTSUM/lm-servers/kill-earthquakes-lm.sh
echo "rm $CUTTSUM/lm-servers/kill-earthquakes-lm.sh" >> $CUTTSUM/lm-servers/kill-earthquakes-lm.sh

