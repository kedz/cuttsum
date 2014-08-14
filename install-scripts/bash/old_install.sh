source $TREC_VENV/bin/activate
DATA=$TREC/data2
LM_INPUT=$DATA/lm_input
LM=$DATA/lm
LM_INPUT_LOG=$TREC/logs/lm_input.log
WGET_LOG=$TREC/logs/wgets.log
if [ ! -d $TREC/logs ]; then
    mkdir -p $TREC/logs
fi

if [ ! -d $DATA ]; then
    echo "Making data directory: $DATA"
    mkdir -p $DATA
fi

EVENTS=$DATA/events
DIRLISTS="${DATA}/dir-lists"
SC2014_SERIF_ONLY_XZ="kba-streamcorpus-2014-v0_3_0-serif-only.s3-paths.txt.xz"
SC2014_SERIF_ONLY="kba-streamcorpus-2014-v0_3_0-serif-only.s3-paths.txt"
SC2014_TS_XZ="streamcorpus-2014-v0_3_0-ts-filtered.s3-paths.txt.xz"
SC2014_TS="streamcorpus-2014-v0_3_0-ts-filtered.s3-paths.txt"
SC2014_SO_TS13="kba-streamcorpus-2014-v0_3_0-serif-only.s3-paths.ts-2013-filtered.txt"

#echo "Downloading hour url-lists for TREC 2014 v3 StreamCorpus"
#echo "========================================================"
if [ ! -d $DIRLISTS ]; then
    echo "Making dir-lists directory: $DIRLISTS"
    mkdir -p $DIRLISTS
fi

cd $DIRLISTS
if [ ! -f $DIRLISTS/$SC2014_SERIF_ONLY ]; then
    if [ ! -f $DIRLISTS/$SC2014_SERIF_ONLY ]; then
        echo "Downloading TREC KBA 2014 url list."
        wget http://s3.amazonaws.com/aws-publicdatasets/trec/kba/$SC2014_SERIF_ONLY_XZ -O $DIRLISTS/$SC2014_SERIF_ONLY_XZ 
    fi
    xz --decompress $DIRLISTS/$SC2014_SERIF_ONLY_XZ
fi

if [ ! -f $DIRLISTS/$SC2014_TS ]; then
    if [ ! -f $DIRLISTS/$SC2014_TS_XZ ]; then
        echo "Downloading TREC KBA 2014 TS 2014 Filtered url list."
        wget  http://s3.amazonaws.com/aws-publicdatasets/trec/ts/$SC2014_TS_XZ -O $DIRLISTS/$SC2014_TS_XZ 
    fi
    xz --decompress $DIRLISTS/$SC2014_TS_XZ
fi

#wget http://s3.amazonaws.com/aws-publicdatasets/trec/kba/kba-streamcorpus-2013-v0_2_0/dir-names.txt

#echo
#echo "Dowloading events for 2013 and 2014"
#echo "==================================="

if [ ! -d $EVENTS ]; then
    echo "Making events directory: $EVENTS"
    mkdir -p $EVENTS
fi

#cd $TREC/data/2013-events
if [ ! -f $EVENTS/test-events-2013.xml ]; then
    echo "Downloading 2013 TREC TS events xml"
    wget -nv http://www.trec-ts.org/topics-masked.xml -O $EVENTS/test-events-2013.xml &>$WGET_LOG
fi

if [ ! -f $EVENTS/test-events-2014.xml ]; then
    echo "Downloading 2014 TREC TS events xml"
    wget -nv http://www.trec-ts.org/trec2014-ts-topics-test.xml -O $EVENTS/test-events-2014.xml &>$WGET_LOG
fi

if [ ! -f $DIRLISTS/$SC2014_SO_TS13 ]; then
    echo "Filtering hour list with 2013 TREC TS event data..."
    python -u $CUTTSUM/install-scripts/filter_hours.py -e $EVENTS/test-events-2013.xml -l $DIRLISTS/$SC2014_SERIF_ONLY -f $DIRLISTS/$SC2014_SO_TS13
fi

if [ ! -d $DATA/2013-event-chunks ]; then
    echo "Making chunk directory: $DATA/2013-event-chunks"
    mkdir -p $DATA/2013-event-chunks
    cd $DATA/2013-event-chunks
    cat $DIRLISTS/$SC2014_SO_TS13 | parallel -j 10 'wget -nv --recursive --continue --no-host-directories --no-parent --reject "index.html*" http://s3.amazonaws.com/aws-publicdatasets/trec/kba/kba-streamcorpus-2014-v0_3_0-serif-only/{}' &>$WGET_LOG
    cd $DATA/2013-event-chunks
    mv aws-publicdatasets/trec/kba/kba-streamcorpus-2014-v0_3_0-serif-only/* .
    rm -rf aws-publicdatasets
fi 

#mkdir -p $TREC/data/2013-data
#cd $TREC/data/2013-data
#if [ ! -f nuggets.tsv ]; then
#    wget http://trec.nist.gov/data/tempsumm/2013/nuggets.tsv
#fi

#python -u $CUTTSUM/resource-install-scripts/download_chunks.py -e $TREC/data/2013-events/topics-masked.xml -c $TREC/data/chunks -l $TREC/data/dir-names.txt

#python -u $CUTTSUM/resource-install-scripts/download_wikis.py -w $CUTTSUM/wiki-lists/earthquakes.txt -d $TREC/data/wiki-html/earthquakes


if [ ! -f $LM_INPUT/gigaword_lm_input.txt ]; then
    echo
    echo "Extracting Gigaword text..."
    python -u $CUTTSUM/install-scripts/gigaword2txt.py -g $GIGAWORD_DATA -of $LM_INPUT/gigaword_lm_input.txt -r '(nyt|apw)_eng_(2009|2008|2007|2006|2005|2004|2003|2002|2001|2000|1999)' &>$LM_INPUT_LOG
fi

if [ ! -f $LM_INPUT/gigaword_vocab.txt ]; then
    echo "Generating vocab file"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/gigaword_lm_input.txt -of $LM_INPUT/gigaword_vocab.txt -t 3 
fi

if [ ! -d $LM/gigaword ]; then
    mkdir -p $LM/gigaword
    echo "Training gigaword 3-gram lm model..."
    ngram-count -order 3 -kndiscount -interpolate -text $LM_INPUT/gigaword_lm_input.txt -vocab $LM_INPUT/gigaword_vocab.txt -lm $LM/gigaword/gigaword_3_lm.txt &
    echo "Training gigaword 4-gram lm model..."
    ngram-count -order 4 -kndiscount -interpolate -text $LM_INPUT/gigaword_lm_input.txt -vocab $LM_INPUT/gigaword_vocab.txt -lm $LM/gigaword/gigaword_4_lm.txt &
    echo "Training gigaword 5-gram lm model..."
    ngram-count -order 5 -kndiscount -interpolate -text $LM_INPUT/gigaword_lm_input.txt -vocab $LM_INPUT/gigaword_vocab.txt -lm $LM/gigaword/gigaword_5_lm.txt &
    wait
fi

#python -u $CUTTSUM/resource-install-scripts/compute_background_word_counts.py -c $TREC/data/chunks -d $TREC/data/word_frequencies -n 20 -l $TREC/logs/word_freqs


### Download wiki pages under a category ###

#echo "Downloading Wikipedia Category Page Lists..."

#python -u $CUTTSUM/resource-install-scripts/build_wiki_lists.py -w $TREC/data/wiki-lists/terrorism.txt -c "Category:Terrorism" -m 10 -l $TREC/logs/wp-lists/terrorism.log &
#
#python -u $CUTTSUM/resource-install-scripts/build_wiki_lists.py -w $TREC/data/wiki-lists/weather.txt -c "Category:Weather events" -m 10 -l $TREC/logs/wp-lists/weather.log &
#
#python -u $CUTTSUM/resource-install-scripts/build_wiki_lists.py -w $TREC/data/wiki-lists/accidents.txt -c "Category:Accidents" -m 10 -l $TREC/logs/wp-lists/accidents.log &
#
#python -u $CUTTSUM/resource-install-scripts/build_wiki_lists.py -w $TREC/data/wiki-lists/shootings.txt -c "Category:Mass shootings" -m 10 -l $TREC/logs/wp-lists/shootings.log &
#
#python -u $CUTTSUM/resource-install-scripts/build_wiki_lists.py -w $TREC/data/wiki-lists/earthquakes.txt -c "Category:Earthquakes" -m 10 -l $TREC/logs/wp-lists/earthquakes.log &
#
#python -u $CUTTSUM/resource-install-scripts/build_wiki_lists.py -w $TREC/data/wiki-lists/social_unrest.txt -c "Category:Activism by type" -m 10 -l $TREC/logs/wp-lists/social_unrest.log &
#
#wait
#printf "Completed downloading page lists.
#

#printf "Downloading Wiki Pages...\n"


if [ ! -d $DATA/wiki-html/earthquakes ]; then
    echo "Installing wiki pages from earthquakes list"
    python -u $CUTTSUM/install-scripts/wiki_list2html.py -d $DATA/wiki-html/earthquakes -w $TREC/data/wiki-lists/earthquakes.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/earthquakes &
else 
    echo "Skipping: Installing wiki pages from earthquakes list"
fi

if [ ! -d $DATA/wiki-html/weather ]; then
    echo "Installing wiki pages from weather list"
    python -u $CUTTSUM/install-scripts/wiki_list2html.py -d $DATA/wiki-html/weather -w $TREC/data/wiki-lists/weather.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/weather &
else 
    echo "Skipping: Installing wiki pages from weather list"
fi

if [ ! -d $DATA/wiki-html/terrorism ]; then
    echo "Installing wiki pages from terrorism list"
    python -u $CUTTSUM/install-scripts/wiki_list2html.py -d $DATA/wiki-html/terrorism -w $TREC/data/wiki-lists/terrorism.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/terrorism &
else 
    echo "Skipping: Installing wiki pages from terrorism list"
fi

if [ ! -d $DATA/wiki-html/accidents ]; then
    echo "Installing wiki pages from accidents list"
    python -u $CUTTSUM/install-scripts/wiki_list2html.py -d $DATA/wiki-html/accidents -w $TREC/data/wiki-lists/accidents.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/accidents &
else 
    echo "Skipping: Installing wiki pages from accidents list"
fi

if [ ! -d $DATA/wiki-html/shootings ]; then
    echo "Installing wiki pages from shootings list"
    python -u $CUTTSUM/install-scripts/wiki_list2html.py -d $DATA/wiki-html/shootings -w $TREC/data/wiki-lists/shootings.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/shootings &
else
    echo "Skipping: Installing wiki pages from shootings list"
fi

if [ ! -d $DATA/wiki-html/social_unrest ]; then
    echo "Installing wiki pages from social unrest list"
    python -u $CUTTSUM/install-scripts/wiki_list2html.py -d $DATA/wiki-html/social_unrest -w $TREC/data/wiki-lists/social_unrest.txt -m 10 -p 4 -l $TREC/logs/wp-downloads/social_unrest &
else 
    echo "Skipping: Installing wiki pages from social unrest list"
fi

wait


if [ ! -d $TREC/logs/html2txt ]; then
    mkdir -p $TREC/logs/html2txt
fi

if [ ! -d $DATA/wiki-text-full/earthquakes ]; then
    echo "Extracting earthquakes full text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/earthquakes -t $DATA/wiki-text-full/earthquakes --full &>$TREC/logs/html2txt/earthquakes.full.log &
else
    echo "Skipping: Extracting earthquakes full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/earthquakes ]; then
    echo "Extracting earthquakes abstract text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/earthquakes -t $DATA/wiki-text-abstract/earthquakes --abstract &>$TREC/logs/html2txt/earthquakes.abs.log &
else
    echo "Skipping: Extracting earthquakes abstract text..."
fi

if [ ! -d $DATA/wiki-text-full/weather ]; then
    echo "Extracting weather full text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/weather -t $DATA/wiki-text-full/weather --full &>$TREC/logs/html2txt/weather.full.log &
else
    echo "Skipping: Extracting weather full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/weather ]; then
    echo "Extracting weather abstract text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/weather -t $DATA/wiki-text-abstract/weather --abstract &>$TREC/logs/html2txt/weather.abs.log &
else
    echo "Skipping: Extracting weather abstract text..."
fi

if [ ! -d $DATA/wiki-text-full/terrorism ]; then
    echo "Extracting terrorism full text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/terrorism -t $DATA/wiki-text-full/terrorism --full &>$TREC/logs/html2txt/terrorism.full.log &
else
    echo "Skipping: Extracting terrorism full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/terrorism ]; then
    echo "Extracting terrorism abstract text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/terrorism -t $DATA/wiki-text-abstract/terrorism --abstract &>$TREC/logs/html2txt/terrorism.abs.log &
else
    echo "Skipping: Extracting terrorism abstract text..."
fi

if [ ! -d $DATA/wiki-text-full/accidents ]; then
    echo "Extracting accidents full text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/accidents -t $DATA/wiki-text-full/accidents --full &>$TREC/logs/html2txt/accidents.full.log &
else
    echo "Skipping: Extracting accidents full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/accidents ]; then
    echo "Extracting accidents abstract text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/accidents -t $DATA/wiki-text-abstract/accidents --abstract &>$TREC/logs/html2txt/accidents.abs.log &
else
    echo "Skipping: Extracting accidents abstract text..."
fi

if [ ! -d $DATA/wiki-text-full/shootings ]; then
    echo "Extracting shootings full text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/shootings -t $DATA/wiki-text-full/shootings --full &>$TREC/logs/html2txt/shootings.full.log &
else
    echo "Skipping: Extracting shootings full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/shootings ]; then
    echo "Extracting shootings abstract text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/shootings -t $DATA/wiki-text-abstract/shootings --abstract &>$TREC/logs/html2txt/shootings.abs.log &
else
    echo "Skipping: Extracting shootings abstract text..."
fi

if [ ! -d $DATA/wiki-text-full/social_unrest ]; then
    echo "Extracting social_unrest full text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/social_unrest -t $DATA/wiki-text-full/social_unrest --full &>$TREC/logs/html2txt/social_unrest.full.log &
else
    echo "Skipping: Extracting social_unrest full text..."
fi

if [ ! -d $DATA/wiki-text-abstract/social_unrest ]; then
    echo "Extracting social_unrest abstract text..."
    python -u $CUTTSUM/install-scripts/html2txt.py -d $DATA/wiki-html/social_unrest -t $DATA/wiki-text-abstract/social_unrest --abstract &>$TREC/logs/html2txt/social_unrest.abs.log &
else
    echo "Skipping: Extracting social_unrest abstract text..."
fi

wait


if [ ! -d $TREC/logs/txt2lm ]; then
    mkdir -p $TREC/logs/txt2lm
fi

if [ ! -f $LM_INPUT/earthquakes_wiki_simple_lm_input.txt ]; then
    echo "Building simple earthquake lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-full/earthquakes -of $LM_INPUT/earthquakes_wiki_simple_lm_input.txt &> $TREC/logs/txt2lm/earthquakes.log &
fi

if [ ! -f $LM_INPUT/earthquakes_wiki_abstracts_spl.txt ]; then
    echo "Building earthquake wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/earthquakes -of $LM_INPUT/earthquakes_wiki_abstracts_spl.txt  &> $TREC/logs/txt2lm/earthquakes.abs.spl.log &
fi

if [ ! -f $LM_INPUT/earthquakes_wiki_abstracts_dpl.txt ]; then
    echo "Building earthquake wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/earthquakes -of $LM_INPUT/earthquakes_wiki_abstracts_dpl.txt --dpl &> $TREC/logs/txt2lm/earthquakes.abs.dpl.log &
fi

if [ ! -f $LM_INPUT/weather_wiki_simple_lm_input.txt ]; then
    echo "Building simple weather lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-full/weather -of $LM_INPUT/weather_wiki_simple_lm_input.txt &> $TREC/logs/txt2lm/weather.log &
fi

if [ ! -f $LM_INPUT/weather_wiki_abstracts_spl.txt ]; then
    echo "Building weather wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/weather -of $LM_INPUT/weather_wiki_abstracts_spl.txt  &> $TREC/logs/txt2lm/weather.abs.spl.log &
fi

if [ ! -f $LM_INPUT/weather_wiki_abstracts_dpl.txt ]; then
    echo "Building weather wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/weather -of $LM_INPUT/weather_wiki_abstracts_dpl.txt --dpl &> $TREC/logs/txt2lm/weather.abs.dpl.log &
fi

if [ ! -f $LM_INPUT/terrorism_wiki_simple_lm_input.txt ]; then
    echo "Building simple terrorism lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-full/terrorism -of $LM_INPUT/terrorism_wiki_simple_lm_input.txt &> $TREC/logs/txt2lm/terrorism.log &
fi

if [ ! -f $LM_INPUT/terrorism_wiki_abstracts_spl.txt ]; then
    echo "Building terrorism wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/terrorism -of $LM_INPUT/terrorism_wiki_abstracts_spl.txt  &> $TREC/logs/txt2lm/terrorism.abs.spl.log &
fi

if [ ! -f $LM_INPUT/terrorism_wiki_abstracts_dpl.txt ]; then
    echo "Building terrorism wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/terrorism -of $LM_INPUT/terrorism_wiki_abstracts_dpl.txt --dpl &> $TREC/logs/txt2lm/terrorism.abs.dpl.log &
fi

if [ ! -f $LM_INPUT/accidents_wiki_simple_lm_input.txt ]; then
    echo "Building simple accidents lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-full/accidents -of $LM_INPUT/accidents_wiki_simple_lm_input.txt &> $TREC/logs/txt2lm/accidents.log &
fi

if [ ! -f $LM_INPUT/accidents_wiki_abstracts_spl.txt ]; then
    echo "Building accidents wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/accidents -of $LM_INPUT/accidents_wiki_abstracts_spl.txt  &> $TREC/logs/txt2lm/accidents.abs.spl.log &
fi

if [ ! -f $LM_INPUT/accidents_wiki_abstracts_dpl.txt ]; then
    echo "Building accidents wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/accidents -of $LM_INPUT/accidents_wiki_abstracts_dpl.txt --dpl &> $TREC/logs/txt2lm/accidents.abs.dpl.log &
fi

if [ ! -f $LM_INPUT/shootings_wiki_simple_lm_input.txt ]; then
    echo "Building simple shootings lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-full/shootings -of $LM_INPUT/shootings_wiki_simple_lm_input.txt &> $TREC/logs/txt2lm/shootings.log &
fi

if [ ! -f $LM_INPUT/shootings_wiki_abstracts_spl.txt ]; then
    echo "Building shootings wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/shootings -of $LM_INPUT/shootings_wiki_abstracts_spl.txt  &> $TREC/logs/txt2lm/shootings.abs.spl.log &
fi

if [ ! -f $LM_INPUT/shootings_wiki_abstracts_dpl.txt ]; then
    echo "Building shootings wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/shootings -of $LM_INPUT/shootings_wiki_abstracts_dpl.txt --dpl &> $TREC/logs/txt2lm/shootings.abs.dpl.log &
fi

if [ ! -f $LM_INPUT/social_unrest_wiki_simple_lm_input.txt ]; then
    echo "Building simple social_unrest lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-full/social_unrest -of $LM_INPUT/social_unrest_wiki_simple_lm_input.txt &> $TREC/logs/txt2lm/social_unrest.log &
fi

if [ ! -f $LM_INPUT/social_unrest_wiki_abstracts_spl.txt ]; then
    echo "Building social_unrest wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/social_unrest -of $LM_INPUT/social_unrest_wiki_abstracts_spl.txt  &> $TREC/logs/txt2lm/social_unrest.abs.spl.log &
fi

if [ ! -f $LM_INPUT/social_unrest_wiki_abstracts_dpl.txt ]; then
    echo "Building social_unrest wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/txt2lm_input.py -t $DATA/wiki-text-abstract/social_unrest -of $LM_INPUT/social_unrest_wiki_abstracts_dpl.txt --dpl &> $TREC/logs/txt2lm/social_unrest.abs.dpl.log &
fi

#
#python -u $CUTTSUM/resource-install-scripts/text2lm_input.py -t $TREC/data/wiki-text/weather -of $LM_INPUT/weather_wiki_simple_lm_input &> $TREC/logs/text2lm/weather.log &
#
#python -u $CUTTSUM/resource-install-scripts/text2lm_input.py -t $TREC/data/wiki-text/terrorism -of $LM_INPUT/terrorism_wiki_simple_lm_input &> $TREC/logs/text2lm/terrorism.log &
#
#python -u $CUTTSUM/resource-install-scripts/text2lm_input.py -t $TREC/data/wiki-text/accidents -of $LM_INPUT/accidents_wiki_simple_lm_input &> $TREC/logs/text2lm/accidents.log &
#
#python -u $CUTTSUM/resource-install-scripts/text2lm_input.py -t $TREC/data/wiki-text/shootings -of $LM_INPUT/shootings_wiki_simple_lm_input &> $TREC/logs/text2lm/shootings.log &
#
#python -u $CUTTSUM/resource-install-scripts/text2lm_input.py -t $TREC/data/wiki-text/social_unrest -of $LM_INPUT/social_unrest_wiki_simple_lm_input &> $TREC/logs/text2lm/social_unreat.log &
#
wait
#printf "Finished writing lm input formats.\n"

cat $LM_INPUT/shootings_wiki_simple_lm_input.txt >$LM_INPUT/terrshoot_wiki_simple_lm_input.txt
cat $LM_INPUT/terrorism_wiki_simple_lm_input.txt >>$LM_INPUT/terrshoot_wiki_simple_lm_input.txt

cat $LM_INPUT/shootings_wiki_abstracts_spl.txt >$LM_INPUT/terrshoot_wiki_abstracts_spl.txt
cat $LM_INPUT/terrorism_wiki_abstracts_spl.txt >>$LM_INPUT/terrshoot_wiki_abstracts_spl.txt

cat $LM_INPUT/shootings_wiki_abstracts_dpl.txt >$LM_INPUT/terrshoot_wiki_abstracts_dpl.txt
cat $LM_INPUT/terrorism_wiki_abstracts_dpl.txt >>$LM_INPUT/terrshoot_wiki_abstracts_dpl.txt


if [ ! -f $LM_INPUT/earthquakes_wiki_simple_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/earthquakes_wiki_simple_vocab.txt"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/earthquakes_wiki_simple_lm_input.txt -of $LM_INPUT/earthquakes_wiki_simple_vocab.txt -t 3 &
fi

if [ ! -f $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/earthquakes_wiki_abstracts_spl.txt -of $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt -t 3 &
fi

if [ ! -f $LM_INPUT/weather_wiki_simple_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/weather_wiki_simple_vocab.txt"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/weather_wiki_simple_lm_input.txt -of $LM_INPUT/weather_wiki_simple_vocab.txt -t 3 &
fi

if [ ! -f $LM_INPUT/weather_wiki_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/weather_wiki_abstracts_vocab.txt"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/weather_wiki_abstracts_spl.txt -of $LM_INPUT/weather_wiki_abstracts_vocab.txt -t 3 &
fi

if [ ! -f $LM_INPUT/accidents_wiki_simple_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/accidents_wiki_simple_vocab.txt"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/accidents_wiki_simple_lm_input.txt -of $LM_INPUT/accidents_wiki_simple_vocab.txt -t 3 &
fi

if [ ! -f $LM_INPUT/accidents_wiki_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/accidents_wiki_abstracts_vocab.txt"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/accidents_wiki_abstracts_spl.txt -of $LM_INPUT/accidents_wiki_abstracts_vocab.txt -t 3 &
fi

if [ ! -f $LM_INPUT/terrshoot_wiki_simple_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/terrshoot_wiki_simple_vocab.txt"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/terrshoot_wiki_simple_lm_input.txt -of $LM_INPUT/terrshoot_wiki_simple_vocab.txt -t 3 &
fi

if [ ! -f $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/terrshoot_wiki_abstracts_spl.txt -of $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt -t 3 &
fi

if [ ! -f $LM_INPUT/social_unrest_wiki_simple_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/social_unrest_wiki_simple_vocab.txt"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/social_unrest_wiki_simple_lm_input.txt -of $LM_INPUT/social_unrest_wiki_simple_vocab.txt -t 3 &
fi

if [ ! -f $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt"
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $LM_INPUT/social_unrest_wiki_abstracts_spl.txt -of $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt -t 3 &
fi

if [ ! -f $LM_INPUT/all_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/all_abstracts_vocab.txt"
    IF1=$LM_INPUT/earthquakes_wiki_abstracts_spl.txt
    IF2=$LM_INPUT/weather_wiki_abstracts_spl.txt
    IF3=$LM_INPUT/accidents_wiki_abstracts_spl.txt
    IF4=$LM_INPUT/terrshoot_wiki_abstracts_spl.txt
    IF5=$LM_INPUT/social_unrest_wiki_abstracts_spl.txt
    python -u $CUTTSUM/install-scripts/generate_vocab.py -if $IF1 $IF2 $IF3 $IF4 $IF5 -of $LM_INPUT/all_abstracts_vocab.txt -t 3 &
fi

wait

F1A="${LM_INPUT}/earthquakes_wiki_abstracts_spl.txt"
F1B="${LM_INPUT}/earthquakes_wiki_abstracts_dpl.txt"
F2A="${LM_INPUT}/weather_wiki_abstracts_spl.txt"
F2B="${LM_INPUT}/weather_wiki_abstracts_dpl.txt"
F3A="${LM_INPUT}/accidents_wiki_abstracts_spl.txt"
F3B="${LM_INPUT}/accidents_wiki_abstracts_dpl.txt"
F4A="${LM_INPUT}/terrshoot_wiki_abstracts_spl.txt"
F4B="${LM_INPUT}/terrshoot_wiki_abstracts_dpl.txt"
F5A="${LM_INPUT}/social_unrest_wiki_abstracts_spl.txt"
F5B="${LM_INPUT}/social_unrest_wiki_abstracts_dpl.txt"
FLISTA="$F1A $F2A $F3A $F4A $F5A"
FLISTB="$F1B $F2B $F3B $F4B $F5B"

if [ ! -d $TREC/logs/wtmf ]; then
    mkdir -p $TREC/logs/wtmf
fi

if [ ! -d  $DATA/sentence-sim/all_events_spl_model/ ]; then
    echo "Training sentence sim for all events spl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $FLISTA -d $DATA/sentence-sim/all_events_spl_model/ -v $LM_INPUT/all_abstracts_vocab.txt &>$TREC/logs/wtmf/all_events_spl.log &
else
    echo "Skipping: Training sentence sim for all events spl..."
fi

if [ ! -d  $DATA/sentence-sim/all_events_dpl_model/ ]; then
    echo "Training sentence sim for all events dpl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $FLISTB -d $DATA/sentence-sim/all_events_dpl_model/ -v $LM_INPUT/all_abstracts_vocab.txt  &>$TREC/logs/wtmf/all_events_dpl.log &
else
    echo "Skipping: Training sentence sim for all events dpl..."
fi

if [ ! -d  $DATA/sentence-sim/earthquakes_spl_model/ ]; then
    echo "Training sentence sim for earthquakes spl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $F1A -d $DATA/sentence-sim/earthquakes_spl_model/ -v $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt &>$TREC/logs/wtmf/earthquakes_spl.log &
else
    echo "Skipping: Training sentence sim for earthquakes spl..."
fi

if [ ! -d  $DATA/sentence-sim/earthquakes_dpl_model/ ]; then
    echo "Training sentence sim for earthquakes dpl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $F1B -d $DATA/sentence-sim/earthquakes_dpl_model/ -v $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt  &>$TREC/logs/wtmf/earthquakes_dpl.log &
else
    echo "Skipping: Training sentence sim for earthquakes dpl..."
fi

if [ ! -d  $DATA/sentence-sim/weather_spl_model/ ]; then
    echo "Training sentence sim for weather spl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $F2A -d $DATA/sentence-sim/weather_spl_model/ -v $LM_INPUT/weather_wiki_abstracts_vocab.txt &>$TREC/logs/wtmf/weather_spl.log &
else
    echo "Skipping: Training sentence sim for weather spl..."
fi

if [ ! -d  $DATA/sentence-sim/weather_dpl_model/ ]; then
    echo "Training sentence sim for weather dpl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $F2B -d $DATA/sentence-sim/weather_dpl_model/ -v $LM_INPUT/weather_wiki_abstracts_vocab.txt  &>$TREC/logs/wtmf/weather_dpl.log &
else
    echo "Skipping: Training sentence sim for weather dpl..."
fi

if [ ! -d  $DATA/sentence-sim/accidents_spl_model/ ]; then
    echo "Training sentence sim for accidents spl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $F3A -d $DATA/sentence-sim/accidents_spl_model/ -v $LM_INPUT/accidents_wiki_abstracts_vocab.txt &>$TREC/logs/wtmf/accidents_spl.log &
else
    echo "Skipping: Training sentence sim for accidents spl..."
fi

if [ ! -d  $DATA/sentence-sim/accidents_dpl_model/ ]; then
    echo "Training sentence sim for accidents dpl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $F3B -d $DATA/sentence-sim/accidents_dpl_model/ -v $LM_INPUT/accidents_wiki_abstracts_vocab.txt  &>$TREC/logs/wtmf/accidents_dpl.log &
else
    echo "Skipping: Training sentence sim for accidents dpl..."
fi

if [ ! -d  $DATA/sentence-sim/terrshoot_spl_model/ ]; then
    echo "Training sentence sim for terrshoot spl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $F4A -d $DATA/sentence-sim/terrshoot_spl_model/ -v $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt &>$TREC/logs/wtmf/terrshoot_spl.log &
else
    echo "Skipping: Training sentence sim for terrshoot spl..."
fi

if [ ! -d  $DATA/sentence-sim/terrshoot_dpl_model/ ]; then
    echo "Training sentence sim for terrshoot dpl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $F4B -d $DATA/sentence-sim/terrshoot_dpl_model/ -v $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt  &>$TREC/logs/wtmf/terrshoot_dpl.log &
else
    echo "Skipping: Training sentence sim for terrshoot dpl..."
fi

if [ ! -d  $DATA/sentence-sim/social_unrest_spl_model/ ]; then
    echo "Training sentence sim for social_unrest spl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $F5A -d $DATA/sentence-sim/social_unrest_spl_model/ -v $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt &>$TREC/logs/wtmf/social_unrest_spl.log &
else
    echo "Skipping: Training sentence sim for social_unrest spl..."
fi

if [ ! -d  $DATA/sentence-sim/social_unrest_dpl_model/ ]; then
    echo "Training sentence sim for social_unrest dpl..."
    python -u $CUTTSUM/install-scripts/train_sentence_similarity.py -i $F5B -d $DATA/sentence-sim/social_unrest_dpl_model/ -v $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt  &>$TREC/logs/wtmf/social_unrest_dpl.log &
else
    echo "Skipping: Training sentence sim for social_unrest dpl..."
fi

wait

#printf "Replacing rare words with unknown token...\n"
#
#python -u $CUTTSUM/resource-install-scripts/replace_unknown.py -if $LM_INPUT/earthquake_wiki_simple_lm_input -of $LM_INPUT/earthquake_wiki_simple_unk_lm_input &
#
#python -u $CUTTSUM/resource-install-scripts/replace_unknown.py -if $LM_INPUT/weather_wiki_simple_lm_input -of $LM_INPUT/weather_wiki_simple_unk_lm_input &
#
#python -u $CUTTSUM/resource-install-scripts/replace_unknown.py -if $LM_INPUT/terrorism_wiki_simple_lm_input -of $LM_INPUT/terrorism_wiki_simple_unk_lm_input &
#
#python -u $CUTTSUM/resource-install-scripts/replace_unknown.py -if $LM_INPUT/accidents_wiki_simple_lm_input -of $LM_INPUT/accidents_wiki_simple_unk_lm_input &
#
#python -u $CUTTSUM/resource-install-scripts/replace_unknown.py -if $LM_INPUT/shootings_wiki_simple_lm_input -of $LM_INPUT/shootings_wiki_simple_unk_lm_input &
#
#python -u $CUTTSUM/resource-install-scripts/replace_unknown.py -if $LM_INPUT/social_unrest_wiki_simple_lm_input -of $LM_INPUT/social_unrest_wiki_simple_unk_lm_input &
#
#wait
#printf "Finished inserting unk tokens.\n"


#### BUILD WIKIPEDIA LMs ###

#printf "Building language model counts...\n"
#

if [ ! -d $LM/domain ]; then
    mkdir -p $LM/domain
fi

if [ ! -f $LM/domain/earthquakes_3.arpa ]; then
    echo "Training trigram earthquakes lm..."
    ngram-count -order 3 -kndiscount -interpolate -text $LM_INPUT/earthquakes_wiki_simple_lm_input.txt -vocab $LM_INPUT/earthquakes_wiki_simple_vocab.txt -lm $LM/domain/earthquakes_3.arpa &
else
    echo "Skipping: Training trigram earthquakes lm..."
fi

if [ ! -f $LM/domain/weather_3.arpa ]; then
    echo "Training trigram weather lm..."
    ngram-count -order 3 -kndiscount -interpolate -text $LM_INPUT/weather_wiki_simple_lm_input.txt -vocab $LM_INPUT/weather_wiki_simple_vocab.txt -lm $LM/domain/weather_3.arpa &
else
    echo "Skipping: Training trigram weather lm..."
fi

if [ ! -f $LM/domain/accidents_3.arpa ]; then
    echo "Training trigram accidents lm..."
    ngram-count -order 3 -kndiscount -interpolate -text $LM_INPUT/accidents_wiki_simple_lm_input.txt -vocab $LM_INPUT/accidents_wiki_simple_vocab.txt -lm $LM/domain/accidents_3.arpa &
else
    echo "Skipping: Training trigram accidents lm..."
fi

if [ ! -f $LM/domain/terrshoot_3.arpa ]; then
    echo "Training trigram terrshoot lm..."
    ngram-count -order 3 -kndiscount -interpolate -text $LM_INPUT/terrshoot_wiki_simple_lm_input.txt -vocab $LM_INPUT/terrshoot_wiki_simple_vocab.txt -lm $LM/domain/terrshoot_3.arpa &
else
    echo "Skipping: Training trigram terrshoot lm..."
fi

if [ ! -f $LM/domain/social_unrest_3.arpa ]; then
    echo "Training trigram social_unrest lm..."
    ngram-count -order 3 -kndiscount -interpolate -text $LM_INPUT/social_unrest_wiki_simple_lm_input.txt -vocab $LM_INPUT/social_unrest_wiki_simple_vocab.txt -lm $LM/domain/social_unrest_3.arpa &
else
    echo "Skipping: Training trigram social_unrest lm..."
fi

wait

#ngram-count -text $LM_INPUT/earthquake_wiki_simple_unk_lm_input -wbdiscount -interpolate -unk -lm $LM/earthquake_wiki_simple_unk.lm &
#ngram-count -text $LM_INPUT/weather_wiki_simple_unk_lm_input -wbdiscount -interpolate -unk -lm $LM/weather_wiki_simple_unk.lm &
#ngram-count -text $LM_INPUT/terrorism_wiki_simple_unk_lm_input -wbdiscount -interpolate -unk -lm $LM/terrorism_wiki_simple_unk.lm &
#ngram-count -text $LM_INPUT/accidents_wiki_simple_unk_lm_input -wbdiscount -interpolate -unk -lm $LM/accidents_wiki_simple_unk.lm &
#ngram-count -text $LM_INPUT/shootings_wiki_simple_unk_lm_input -wbdiscount -interpolate -unk -lm $LM/shootings_wiki_simple_unk.lm &
#ngram-count -text $LM_INPUT/social_unrest_wiki_simple_unk_lm_input -wbdiscount -interpolate -unk -lm $LM/social_unrest_wiki_simple_unk.lm &
#wait
#printf "Finished building ngram counts.\n"
#
#
echo 
echo "Finished installing resources for CU TREC TS."
