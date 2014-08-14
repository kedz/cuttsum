if [ ! -d $TREC/logs/html2txt ]; then
    mkdir -p $TREC/logs/html2txt
fi

if [ ! -d $DATA/wiki-text-full/earthquakes ]; then
    echo "Extracting earthquakes full text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/earthquakes \
        -t $DATA/wiki-text-full/earthquakes --full \
        1>$TREC/logs/html2txt/earthquakes.full.log \
        2>$TREC/logs/html2txt/earthquakes.full.err &
else
    echo "Skipping: Extracting earthquakes full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/earthquakes ]; then
    echo "Extracting earthquakes abstract text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/earthquakes \
        -t $DATA/wiki-text-abstract/earthquakes --abstract \
        1>$TREC/logs/html2txt/earthquakes.abs.log \
        2>$TREC/logs/html2txt/earthquakes.abs.err &
else
    echo "Skipping: Extracting earthquakes abstract text..."
fi

if [ ! -d $DATA/wiki-text-full/weather ]; then
    echo "Extracting weather full text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/weather \
        -t $DATA/wiki-text-full/weather --full \
        1>$TREC/logs/html2txt/weather.full.log \
        2>$TREC/logs/html2txt/weather.full.err &
else
    echo "Skipping: Extracting weather full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/weather ]; then
    echo "Extracting weather abstract text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/weather \
        -t $DATA/wiki-text-abstract/weather --abstract \
        1>$TREC/logs/html2txt/weather.abs.log \
        2>$TREC/logs/html2txt/weather.abs.err &
else
    echo "Skipping: Extracting weather abstract text..."
fi

if [ ! -d $DATA/wiki-text-full/terrorism ]; then
    echo "Extracting terrorism full text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/terrorism \
        -t $DATA/wiki-text-full/terrorism --full \
        1>$TREC/logs/html2txt/terrorism.full.log \
        2>$TREC/logs/html2txt/terrorism.full.err &
else
    echo "Skipping: Extracting terrorism full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/terrorism ]; then
    echo "Extracting terrorism abstract text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/terrorism \
        -t $DATA/wiki-text-abstract/terrorism --abstract \
        1>$TREC/logs/html2txt/terrorism.abs.log \
        2>$TREC/logs/html2txt/terrorism.abs.err &
else
    echo "Skipping: Extracting terrorism abstract text..."
fi

if [ ! -d $DATA/wiki-text-full/accidents ]; then
    echo "Extracting accidents full text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/accidents \
        -t $DATA/wiki-text-full/accidents --full \
        1>$TREC/logs/html2txt/accidents.full.log \
        2>$TREC/logs/html2txt/accidents.full.err &
else
    echo "Skipping: Extracting accidents full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/accidents ]; then
    echo "Extracting accidents abstract text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/accidents \
        -t $DATA/wiki-text-abstract/accidents --abstract \
        1>$TREC/logs/html2txt/accidents.abs.log \
        2>$TREC/logs/html2txt/accidents.abs.err &
else
    echo "Skipping: Extracting accidents abstract text..."
fi

if [ ! -d $DATA/wiki-text-full/shootings ]; then
    echo "Extracting shootings full text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/shootings \
        -t $DATA/wiki-text-full/shootings --full \
        1>$TREC/logs/html2txt/shootings.full.log \
        2>$TREC/logs/html2txt/shootings.full.err &
else
    echo "Skipping: Extracting shootings full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/shootings ]; then
    echo "Extracting shootings abstract text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/shootings \
        -t $DATA/wiki-text-abstract/shootings --abstract \
        1>$TREC/logs/html2txt/shootings.abs.log \
        2>$TREC/logs/html2txt/shootings.abs.err &
else
    echo "Skipping: Extracting shootings abstract text..."
fi

if [ ! -d $DATA/wiki-text-full/social_unrest ]; then
    echo "Extracting social_unrest full text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/social_unrest \
        -t $DATA/wiki-text-full/social_unrest --full \
        1>$TREC/logs/html2txt/social_unrest.full.log \
        2>$TREC/logs/html2txt/social_unrest.full.err &
else
    echo "Skipping: Extracting social_unrest full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/social_unrest ]; then
    echo "Extracting social_unrest abstract text..."
    python -u $CUTTSUM/install-scripts/python/html2txt.py \
        -d $DATA/wiki-html/social_unrest \
        -t $DATA/wiki-text-abstract/social_unrest --abstract \
        1>$TREC/logs/html2txt/social_unrest.abs.log \
        2>$TREC/logs/html2txt/social_unrest.abs.err &
else
    echo "Skipping: Extracting social_unrest abstract text..."
fi

wait
