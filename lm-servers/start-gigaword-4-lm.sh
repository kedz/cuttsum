ngram -lm $DATA/lm/gigaword/gigaword_4.arpa -unk \
    -order 4 \
    -vocab $DATA/lm_input/gigaword_vocab.txt \
    -server-port 9906 &
PID=$!
DIR=`pwd`
echo "kill $PID" > $CUTTSUM/lm-servers/kill-gigaword-4-lm.sh
echo "rm $CUTTSUM/lm-servers/kill-gigaword-4-lm.sh" >> $CUTTSUM/lm-servers/kill-gigaword-4-lm.sh

