if [ ! -d $DATA/wiki-html/earthquakes ]; then
    echo "Installing wiki pages from earthquakes list"
    python -u $CUTTSUM/install-scripts/python/wiki_list2html.py -d $DATA/wiki-html/earthquakes -w $TREC/data/wiki-lists/earthquakes.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/earthquakes &
else 
    echo "Skipping: Installing wiki pages from earthquakes list"
fi

if [ ! -d $DATA/wiki-html/weather ]; then
    echo "Installing wiki pages from weather list"
    python -u $CUTTSUM/install-scripts/python/wiki_list2html.py -d $DATA/wiki-html/weather -w $TREC/data/wiki-lists/weather.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/weather &
else 
    echo "Skipping: Installing wiki pages from weather list"
fi

if [ ! -d $DATA/wiki-html/terrorism ]; then
    echo "Installing wiki pages from terrorism list"
    python -u $CUTTSUM/install-scripts/python/wiki_list2html.py -d $DATA/wiki-html/terrorism -w $TREC/data/wiki-lists/terrorism.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/terrorism &
else 
    echo "Skipping: Installing wiki pages from terrorism list"
fi

if [ ! -d $DATA/wiki-html/accidents ]; then
    echo "Installing wiki pages from accidents list"
    python -u $CUTTSUM/install-scripts/python/wiki_list2html.py -d $DATA/wiki-html/accidents -w $TREC/data/wiki-lists/accidents.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/accidents &
else 
    echo "Skipping: Installing wiki pages from accidents list"
fi

if [ ! -d $DATA/wiki-html/shootings ]; then
    echo "Installing wiki pages from shootings list"
    python -u $CUTTSUM/install-scripts/python/wiki_list2html.py -d $DATA/wiki-html/shootings -w $TREC/data/wiki-lists/shootings.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/shootings &
else
    echo "Skipping: Installing wiki pages from shootings list"
fi

if [ ! -d $DATA/wiki-html/social_unrest ]; then
    echo "Installing wiki pages from social unrest list"
    python -u $CUTTSUM/install-scripts/python/wiki_list2html.py -d $DATA/wiki-html/social_unrest -w $TREC/data/wiki-lists/social_unrest.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/social_unrest &
else 
    echo "Skipping: Installing wiki pages from social unrest list"
fi

wait

