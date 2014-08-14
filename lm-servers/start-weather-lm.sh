ngram -lm $DATA/lm/domain/weather_3.arpa -unk \
    -vocab $DATA/lm_input/weather_wiki_simple_vocab.txt \
    -server-port 9901 &
PID=$!
DIR=`pwd`
echo "kill $PID" > $CUTTSUM/lm-servers/kill-weather-lm.sh
echo "rm $CUTTSUM/lm-servers/kill-weather-lm.sh" >> $CUTTSUM/lm-servers/kill-weather-lm.sh

