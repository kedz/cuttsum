ngram -lm $DATA/lm/domain/accidents_3.arpa -unk \
    -vocab $DATA/lm_input/accidents_wiki_simple_vocab.txt \
    -server-port 9903 &
PID=$!
DIR=`pwd`
echo "kill $PID" > $CUTTSUM/lm-servers/kill-accidents-lm.sh
echo "rm $CUTTSUM/lm-servers/kill-accidents-lm.sh" >> $CUTTSUM/lm-servers/kill-accidents-lm.sh

