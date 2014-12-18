ngram -lm $DATA/lm/gigaword/gigaword_5.arpa -unk \
    -order 5 \
    -vocab $DATA/lm_input/gigaword_vocab.txt \
    -server-port 9907 &
PID=$!
DIR=`pwd`
echo "kill $PID" > $CUTTSUM/lm-servers/kill-gigaword-5-lm.sh
echo "rm $CUTTSUM/lm-servers/kill-gigaword-5-lm.sh" >> $CUTTSUM/lm-servers/kill-gigaword-5-lm.sh

