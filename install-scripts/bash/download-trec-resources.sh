EVENTS=$DATA/events
NUGGETS=$DATA/nuggets
DIRLISTS="${DATA}/dir-lists"
SC2014_SERIF_ONLY_XZ="kba-streamcorpus-2014-v0_3_0-serif-only.s3-paths.txt.xz"
SC2014_SERIF_ONLY="kba-streamcorpus-2014-v0_3_0-serif-only.s3-paths.txt"
SC2014_TS_XZ="streamcorpus-2014-v0_3_0-ts-filtered.s3-paths.txt.xz"
SC2014_TS="streamcorpus-2014-v0_3_0-ts-filtered.s3-paths.txt"
SC2014_SO_TS13="kba-streamcorpus-2014-v0_3_0-serif-only.s3-paths.ts-2013-filtered.txt"

if [ ! -d $DIRLISTS ]; then
    echo "Making dir-lists directory: $DIRLISTS"
    mkdir -p $DIRLISTS
else
    echo "Skipping: Making dir-lists directory: $DIRLISTS"
fi

cd $DIRLISTS
if [ ! -f $DIRLISTS/$SC2014_SERIF_ONLY ]; then
    if [ ! -f $DIRLISTS/$SC2014_SERIF_ONLY ]; then
        echo "Downloading TREC KBA 2014 url list."
        wget http://s3.amazonaws.com/aws-publicdatasets/trec/kba/$SC2014_SERIF_ONLY_XZ \
            -O $DIRLISTS/$SC2014_SERIF_ONLY_XZ 
    fi
    xz --decompress $DIRLISTS/$SC2014_SERIF_ONLY_XZ
else
    echo "Skipping: Downloading TREC KBA 2014 url list."
fi

if [ ! -f $DIRLISTS/$SC2014_TS ]; then
    if [ ! -f $DIRLISTS/$SC2014_TS_XZ ]; then
        echo "Downloading TREC KBA 2014 TS 2014 Filtered url list."
        wget  http://s3.amazonaws.com/aws-publicdatasets/trec/ts/$SC2014_TS_XZ \
            -O $DIRLISTS/$SC2014_TS_XZ 
    fi
    xz --decompress $DIRLISTS/$SC2014_TS_XZ
else
    echo "Skipping: Downloading TREC KBA 2014 TS 2014 Filtered url list."
fi

if [ ! -d $NUGGETS ]; then
    echo "Making nuggets directory: $NUGGETS"
    mkdir -p $NUGGETS
else
    echo "Skipping: Making nuggets directory: $NUGGETS"
fi

if [ ! -f $NUGGETS/test-nuggets-2013.xml ]; then
    echo "Downloading 2013 TREC TS nuggets xml"
    wget -nv http://trec.nist.gov/data/tempsumm/2013/nuggets.tsv \
        -O $NUGGETS/test-nuggets-2013.xml \
        &>$WGET_LOG
else
    echo "Skipping: Downloading 2013 TREC TS nuggets xml"
fi

if [ ! -d $EVENTS ]; then
    echo "Making events directory: $EVENTS"
    mkdir -p $EVENTS
else
    echo "Skipping: Making events directory: $EVENTS"
fi

if [ ! -f $EVENTS/test-events-2013.xml ]; then
    echo "Downloading 2013 TREC TS events xml"
    wget -nv http://www.trec-ts.org/topics-masked.xml \
        -O $EVENTS/test-events-2013.xml \
        &>$WGET_LOG
else
    echo "Skipping: Downloading 2013 TREC TS events xml"
fi

if [ ! -f $EVENTS/test-events-2014.xml ]; then
    echo "Downloading 2014 TREC TS events xml"
    wget -nv http://www.trec-ts.org/trec2014-ts-topics-test.xml \
        -O $EVENTS/test-events-2014.xml \
        &>$WGET_LOG
else
    echo "Skipping: Downloading 2014 TREC TS events xml"
fi

if [ ! -f $DIRLISTS/$SC2014_SO_TS13 ]; then
    echo "Filtering hour list with 2013 TREC TS event data..."
    python -u $CUTTSUM/install-scripts/python/filter_hours.py \
        -e $EVENTS/test-events-2013.xml \
        -l $DIRLISTS/$SC2014_SERIF_ONLY \
        -f $DIRLISTS/$SC2014_SO_TS13
else
    echo "Skipping: Filtering hour list with 2013 TREC TS event data..."
fi

if [ ! -d $DATA/2013-event-chunks ]; then
    echo "Downloading KBA chunks to: $DATA/2013-event-chunks"
    mkdir -p $DATA/2013-event-chunks
    cd $DATA/2013-event-chunks
    WGETCMD="wget -nv --recursive --continue --no-host-directories "
    WGETCMD="${WGETCMD}--no-parent --reject \"index.html*\" "
    WGETCMD="${WGETCMD}http://s3.amazonaws.com/aws-publicdatasets/trec/kba/"
    WGETCMD="${WGETCMD}kba-streamcorpus-2014-v0_3_0-serif-only/{}" 
    cat $DIRLISTS/$SC2014_SO_TS13 | parallel -j 10 $WGETCMD &>$WGET_LOG
    cd $DATA/2013-event-chunks
    mv aws-publicdatasets/trec/kba/kba-streamcorpus-2014-v0_3_0-serif-only/* .
    rm -rf aws-publicdatasets


else
    echo "Skipping: Downloading KBA chunks to: $DATA/2013-event-chunks"
fi

if [ ! -d $DATA/2014-event-chunks-filtered-ts ]; then
    echo "Downloading KBA 2014 filtered ts chunks to: $DATA/2014-event-chunks-filtered-ts"
    mkdir -p $DATA/2014-event-chunks-filtered-ts
    cd $DATA/2014-event-chunks-filtered-ts
    WGETCMD="wget -nv --recursive --continue --no-host-directories "
    WGETCMD="${WGETCMD}--no-parent --reject \"index.html*\" http{}"
    cat $DIRLISTS/$SC2014_TS | grep -i news | cut -d ':' -f3 \
        | sed 's/\/\//:\/\/s3.amazonaws.com\//g' \
        | parallel -j 10 $WGETCMD &>$WGET_LOG
    cd $DATA/2014-event-chunks-filtered-ts
    mv aws-publicdatasets/trec/ts/streamcorpus-2014-v0_3_0-ts-filtered/* .
    rm -rf aws-publicdatasets
else
    echo "Skipping: Downloading KBA 2014 filtered ts chunks to: $DATA/2014-event-chunks-filtered-ts"
fi

