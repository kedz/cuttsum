
if [ ! -f $CUTTSUM/lm-servers/kill-earthquakes-lm.sh ]; then
    echo "Starting earthquake lm server @9900"
    bash $CUTTSUM/lm-servers/start-earthquakes-lm.sh
else
    echo "Skipping: Starting earthquakes lm server @9900"
fi

if [ ! -f $CUTTSUM/lm-servers/kill-weather-lm.sh ]; then
    echo "Starting weather lm server @9901"
    bash $CUTTSUM/lm-servers/start-weather-lm.sh
else
    echo "Skipping: Starting weather lm server @9901"
fi

if [ ! -f $CUTTSUM/lm-servers/kill-terrshoot-lm.sh ]; then
    echo "Starting terrshoot lm server @9902"
    bash $CUTTSUM/lm-servers/start-terrshoot-lm.sh
else
    echo "Skipping: Starting terrshoot lm server @9902"
fi

if [ ! -f $CUTTSUM/lm-servers/kill-accidents-lm.sh ]; then
    echo "Starting accidents lm server @9903"
    bash $CUTTSUM/lm-servers/start-accidents-lm.sh
else
    echo "Skipping: Starting accidents lm server @9903"
fi

if [ ! -f $CUTTSUM/lm-servers/kill-social_unrest-lm.sh ]; then
    echo "Starting social_unrest lm server @9904"
    bash $CUTTSUM/lm-servers/start-social_unrest-lm.sh
else
    echo "Skipping: Starting social_unrest lm server @9904"
fi

if [ ! -f $CUTTSUM/lm-servers/kill-gigaword-3-lm.sh ]; then
    echo "Starting gigaword 3-gram lm server @9905"
    bash $CUTTSUM/lm-servers/start-gigaword-3-lm.sh
else
    echo "Skipping: Starting gigaword 3-gram lm server @9905"
fi

if [ ! -f $CUTTSUM/lm-servers/kill-gigaword-4-lm.sh ]; then
    echo "Starting gigaword 4-gram lm server @9906"
    bash $CUTTSUM/lm-servers/start-gigaword-4-lm.sh
else
    echo "Skipping: Starting gigaword 4-gram lm server @9906"
fi

if [ ! -f $CUTTSUM/lm-servers/kill-gigaword-5-lm.sh ]; then
    echo "Starting gigaword 5-gram lm server @9907"
    bash $CUTTSUM/lm-servers/start-gigaword-5-lm.sh
else
    echo "Skipping: Starting gigaword 5-gram lm server @9907"
fi

