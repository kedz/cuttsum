ngram -lm $DATA/lm/gigaword/gigaword_3.arpa -unk \
    -vocab $DATA/lm_input/gigaword_vocab.txt \
    -server-port 9905 &
PID=$!
DIR=`pwd`
echo "kill $PID" > $CUTTSUM/lm-servers/kill-gigaword-3-lm.sh
echo "rm $CUTTSUM/lm-servers/kill-gigaword-3-lm.sh" >> $CUTTSUM/lm-servers/kill-gigaword-3-lm.sh

