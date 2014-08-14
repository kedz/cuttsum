ngram -lm $DATA/lm/domain/social_unrest_3.arpa -unk \
    -vocab $DATA/lm_input/social_unrest_wiki_simple_vocab.txt \
    -server-port 9904 &
PID=$!
DIR=`pwd`
echo "kill $PID" > $CUTTSUM/lm-servers/kill-social_unrest-lm.sh
echo "rm $CUTTSUM/lm-servers/kill-social_unrest-lm.sh" >> $CUTTSUM/lm-servers/kill-social_unrest-lm.sh

