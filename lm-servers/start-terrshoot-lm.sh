ngram -lm $DATA/lm/domain/terrshoot_3.arpa -unk \
    -vocab $DATA/lm_input/terrshoot_wiki_simple_vocab.txt \
    -server-port 9902 &
PID=$!
DIR=`pwd`
echo "kill $PID" > $CUTTSUM/lm-servers/kill-terrshoot-lm.sh
echo "rm $CUTTSUM/lm-servers/kill-terrshoot-lm.sh" >> $CUTTSUM/lm-servers/kill-terrshoot-lm.sh

