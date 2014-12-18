if [ ! -d $TREC/logs/txt2lm ]; then
    mkdir -p $TREC/logs/txt2lm
fi

if [ ! -f $LM_INPUT/earthquakes_wiki_simple_lm_input.txt ]; then
    echo "Building simple earthquake lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-full/earthquakes \
        -of $LM_INPUT/earthquakes_wiki_simple_lm_input.txt \
        1>$TREC/logs/txt2lm/earthquakes.log \
        2>$TREC/logs/txt2lm/earthquakes.err &
else
    echo "Skipping: Building simple earthquake lm input..."
fi

if [ ! -f $LM_INPUT/earthquakes_wiki_abstracts_spl.txt ]; then
    echo "Building earthquake wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/earthquakes \
        -of $LM_INPUT/earthquakes_wiki_abstracts_spl.txt \
        1>$TREC/logs/txt2lm/earthquakes.abs.spl.log \
        2>$TREC/logs/txt2lm/earthquakes.abs.spl.err &
else
    echo "Skipping: Building earthquake wiki abstracts spl lm input..."
fi

if [ ! -f $LM_INPUT/earthquakes_wiki_abstracts_dpl.txt ]; then
    echo "Building earthquake wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/earthquakes \
        -of $LM_INPUT/earthquakes_wiki_abstracts_dpl.txt --dpl \
        1>$TREC/logs/txt2lm/earthquakes.abs.dpl.log \
        2>$TREC/logs/txt2lm/earthquakes.abs.dpl.err &
else
    echo "Skipping: Building earthquake wiki abstracts dpl lm input..."
fi

if [ ! -f $LM_INPUT/weather_wiki_simple_lm_input.txt ]; then
    echo "Building simple weather lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-full/weather \
        -of $LM_INPUT/weather_wiki_simple_lm_input.txt \
        1>$TREC/logs/txt2lm/weather.log \
        2>$TREC/logs/txt2lm/weather.err &
else
    echo "Skipping: Building simple weather lm input..."
fi

if [ ! -f $LM_INPUT/weather_wiki_abstracts_spl.txt ]; then
    echo "Building weather wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/weather \
        -of $LM_INPUT/weather_wiki_abstracts_spl.txt \
        1>$TREC/logs/txt2lm/weather.abs.spl.log \
        2>$TREC/logs/txt2lm/weather.abs.spl.err &
else
    echo "Skipping: Building weather wiki abstracts spl lm input..."
fi

if [ ! -f $LM_INPUT/weather_wiki_abstracts_dpl.txt ]; then
    echo "Building weather wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/weather \
        -of $LM_INPUT/weather_wiki_abstracts_dpl.txt --dpl \
        1>$TREC/logs/txt2lm/weather.abs.dpl.log \
        2>$TREC/logs/txt2lm/weather.abs.dpl.err &
else
    echo "Skipping: Building weather wiki abstracts dpl lm input..."
fi

if [ ! -f $LM_INPUT/terrorism_wiki_simple_lm_input.txt ]; then
    echo "Building simple terrorism lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-full/terrorism \
        -of $LM_INPUT/terrorism_wiki_simple_lm_input.txt \
        1>$TREC/logs/txt2lm/terrorism.log \
        2>$TREC/logs/txt2lm/terrorism.err &
else
    echo "Skipping: Building simple terrorism lm input..."
fi

if [ ! -f $LM_INPUT/terrorism_wiki_abstracts_spl.txt ]; then
    echo "Building terrorism wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/terrorism \
        -of $LM_INPUT/terrorism_wiki_abstracts_spl.txt \
        1>$TREC/logs/txt2lm/terrorism.abs.spl.log \
        2>$TREC/logs/txt2lm/terrorism.abs.spl.err &
else
    echo "Skipping: Building terrorism wiki abstracts spl lm input..."
fi

if [ ! -f $LM_INPUT/terrorism_wiki_abstracts_dpl.txt ]; then
    echo "Building terrorism wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/terrorism \
        -of $LM_INPUT/terrorism_wiki_abstracts_dpl.txt --dpl \
        1>$TREC/logs/txt2lm/terrorism.abs.dpl.log \
        2>$TREC/logs/txt2lm/terrorism.abs.dpl.err &
else
    echo "Skipping: Building terrorism wiki abstracts dpl lm input..."
fi

if [ ! -f $LM_INPUT/accidents_wiki_simple_lm_input.txt ]; then
    echo "Building simple accidents lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py -t \
        $DATA/wiki-text-full/accidents \
        -of $LM_INPUT/accidents_wiki_simple_lm_input.txt \
        1>$TREC/logs/txt2lm/accidents.log \
        2>$TREC/logs/txt2lm/accidents.err &
else
    echo "Skipping: Building simple accidents lm input..."
fi

if [ ! -f $LM_INPUT/accidents_wiki_abstracts_spl.txt ]; then
    echo "Building accidents wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/accidents \
        -of $LM_INPUT/accidents_wiki_abstracts_spl.txt \
        1>$TREC/logs/txt2lm/accidents.abs.spl.log \
        2>$TREC/logs/txt2lm/accidents.abs.spl.err &
else
    echo "Skipping: Building accidents wiki abstracts spl lm input..."
fi

if [ ! -f $LM_INPUT/accidents_wiki_abstracts_dpl.txt ]; then
    echo "Building accidents wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/accidents \
        -of $LM_INPUT/accidents_wiki_abstracts_dpl.txt --dpl \
        1>$TREC/logs/txt2lm/accidents.abs.dpl.log \
        2>$TREC/logs/txt2lm/accidents.abs.dpl.err &
else
    echo "Skipping: Building accidents wiki abstracts dpl lm input..."
fi

if [ ! -f $LM_INPUT/shootings_wiki_simple_lm_input.txt ]; then
    echo "Building simple shootings lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-full/shootings \
        -of $LM_INPUT/shootings_wiki_simple_lm_input.txt \
        1>$TREC/logs/txt2lm/shootings.log \
        2>$TREC/logs/txt2lm/shootings.err &
else
    echo "Skipping: Building simple shootings lm input..."
fi

if [ ! -f $LM_INPUT/shootings_wiki_abstracts_spl.txt ]; then
    echo "Building shootings wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/shootings \
        -of $LM_INPUT/shootings_wiki_abstracts_spl.txt \
        1>$TREC/logs/txt2lm/shootings.abs.spl.log \
        2>$TREC/logs/txt2lm/shootings.abs.spl.err &
else
    echo "Skipping: Building shootings wiki abstracts spl lm input..."
fi

if [ ! -f $LM_INPUT/shootings_wiki_abstracts_dpl.txt ]; then
    echo "Building shootings wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/shootings \
        -of $LM_INPUT/shootings_wiki_abstracts_dpl.txt --dpl \
        1>$TREC/logs/txt2lm/shootings.abs.dpl.log \
        2>$TREC/logs/txt2lm/shootings.abs.dpl.err &
else
    echo "Skipping: Building shootings wiki abstracts dpl lm input..."
fi

if [ ! -f $LM_INPUT/social_unrest_wiki_simple_lm_input.txt ]; then
    echo "Building simple social_unrest lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-full/social_unrest \
        -of $LM_INPUT/social_unrest_wiki_simple_lm_input.txt \
        1>$TREC/logs/txt2lm/social_unrest.log \
        2>$TREC/logs/txt2lm/social_unrest.err &
else
    echo "Skipping: Building simple social_unrest lm input..."
fi

if [ ! -f $LM_INPUT/social_unrest_wiki_abstracts_spl.txt ]; then
    echo "Building social_unrest wiki abstracts spl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/social_unrest \
        -of $LM_INPUT/social_unrest_wiki_abstracts_spl.txt \
        1>$TREC/logs/txt2lm/social_unrest.abs.spl.log \
        2>$TREC/logs/txt2lm/social_unrest.abs.spl.err &
else
    echo "Skipping: Building social_unrest wiki abstracts spl lm input..."
fi

if [ ! -f $LM_INPUT/social_unrest_wiki_abstracts_dpl.txt ]; then
    echo "Building social_unrest wiki abstracts dpl lm input..."
    python -u $CUTTSUM/install-scripts/python/txt2lm_input.py \
        -t $DATA/wiki-text-abstract/social_unrest \
        -of $LM_INPUT/social_unrest_wiki_abstracts_dpl.txt --dpl \
        1>$TREC/logs/txt2lm/social_unrest.abs.dpl.log \
        2>$TREC/logs/txt2lm/social_unrest.abs.dpl.err &
else
    echo "Skipping: Building social_unrest wiki abstracts dpl lm input..."
fi

wait

if [ ! -f $LM_INPUT/terrshoot_wiki_simple_lm_input.txt ]; then
    echo "Merging terrorism and shootings wiki simple lm inputs"
    cat $LM_INPUT/shootings_wiki_simple_lm_input.txt >$LM_INPUT/terrshoot_wiki_simple_lm_input.txt
    cat $LM_INPUT/terrorism_wiki_simple_lm_input.txt >>$LM_INPUT/terrshoot_wiki_simple_lm_input.txt
else
    echo "Skipping: Merging terrorism and shootings wiki simple lm inputs"
fi

if [ ! -f $LM_INPUT/terrshoot_wiki_abstracts_spl.txt ]; then
    echo "Merging terrorism and shootings wiki abstracts spl"
    cat $LM_INPUT/shootings_wiki_abstracts_spl.txt >$LM_INPUT/terrshoot_wiki_abstracts_spl.txt
    cat $LM_INPUT/terrorism_wiki_abstracts_spl.txt >>$LM_INPUT/terrshoot_wiki_abstracts_spl.txt
else
    echo "Skipping: Merging terrorism and shootings wiki abstracts spl"
fi

if [ ! -f $LM_INPUT/terrshoot_wiki_abstracts_dpl.txt ]; then
    echo "Merging terrorism and shootings wiki abstracts dpl"
    cat $LM_INPUT/shootings_wiki_abstracts_dpl.txt >$LM_INPUT/terrshoot_wiki_abstracts_dpl.txt
    cat $LM_INPUT/terrorism_wiki_abstracts_dpl.txt >>$LM_INPUT/terrshoot_wiki_abstracts_dpl.txt
else
    echo "Skipping: Merging terrorism and shootings wiki abstracts dpl"
fi

if [ ! -f $LM_INPUT/earthquakes_wiki_simple_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/earthquakes_wiki_simple_vocab.txt"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/earthquakes_wiki_simple_lm_input.txt \
        -of $LM_INPUT/earthquakes_wiki_simple_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/earthquakes_wiki_simple_vocab.txt"
fi

if [ ! -f $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/earthquakes_wiki_abstracts_spl.txt \
        -of $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/earthquakes_wiki_abstracts_vocab.txt"
fi

if [ ! -f $LM_INPUT/weather_wiki_simple_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/weather_wiki_simple_vocab.txt"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/weather_wiki_simple_lm_input.txt \
        -of $LM_INPUT/weather_wiki_simple_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/weather_wiki_simple_vocab.txt"
fi

if [ ! -f $LM_INPUT/weather_wiki_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/weather_wiki_abstracts_vocab.txt"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/weather_wiki_abstracts_spl.txt \
        -of $LM_INPUT/weather_wiki_abstracts_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/weather_wiki_abstracts_vocab.txt"
fi

if [ ! -f $LM_INPUT/accidents_wiki_simple_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/accidents_wiki_simple_vocab.txt"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/accidents_wiki_simple_lm_input.txt \
        -of $LM_INPUT/accidents_wiki_simple_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/accidents_wiki_simple_vocab.txt"
fi

if [ ! -f $LM_INPUT/accidents_wiki_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/accidents_wiki_abstracts_vocab.txt"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/accidents_wiki_abstracts_spl.txt \
        -of $LM_INPUT/accidents_wiki_abstracts_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/accidents_wiki_abstracts_vocab.txt"
fi

if [ ! -f $LM_INPUT/terrshoot_wiki_simple_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/terrshoot_wiki_simple_vocab.txt"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/terrshoot_wiki_simple_lm_input.txt \
        -of $LM_INPUT/terrshoot_wiki_simple_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/terrshoot_wiki_simple_vocab.txt"
fi

if [ ! -f $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/terrshoot_wiki_abstracts_spl.txt \
        -of $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/terrshoot_wiki_abstracts_vocab.txt"
fi

if [ ! -f $LM_INPUT/social_unrest_wiki_simple_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/social_unrest_wiki_simple_vocab.txt"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/social_unrest_wiki_simple_lm_input.txt \
        -of $LM_INPUT/social_unrest_wiki_simple_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/social_unrest_wiki_simple_vocab.txt"
fi

if [ ! -f $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt"
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $LM_INPUT/social_unrest_wiki_abstracts_spl.txt \
        -of $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/social_unrest_wiki_abstracts_vocab.txt"
fi

if [ ! -f $LM_INPUT/all_abstracts_vocab.txt ]; then
    echo "Generating vocab file: $LM_INPUT/all_abstracts_vocab.txt"
    IF1=$LM_INPUT/earthquakes_wiki_abstracts_spl.txt
    IF2=$LM_INPUT/weather_wiki_abstracts_spl.txt
    IF3=$LM_INPUT/accidents_wiki_abstracts_spl.txt
    IF4=$LM_INPUT/terrshoot_wiki_abstracts_spl.txt
    IF5=$LM_INPUT/social_unrest_wiki_abstracts_spl.txt
    python -u $CUTTSUM/install-scripts/python/generate_vocab.py \
        -if $IF1 $IF2 $IF3 $IF4 $IF5 \
        -of $LM_INPUT/all_abstracts_vocab.txt -t 3 &
else
    echo "Skipping: Generating vocab file: $LM_INPUT/all_abstracts_vocab.txt"
fi

wait

if [ ! -d $LM/domain ]; then
    mkdir -p $LM/domain
fi

if [ ! -f $LM/domain/earthquakes_3.arpa ]; then
    echo "Training trigram earthquakes lm..."
    ngram-count -order 3 -kndiscount -interpolate \
        -text $LM_INPUT/earthquakes_wiki_simple_lm_input.txt \
        -vocab $LM_INPUT/earthquakes_wiki_simple_vocab.txt \
        -lm $LM/domain/earthquakes_3.arpa &
else
    echo "Skipping: Training trigram earthquakes lm..."
fi

if [ ! -f $LM/domain/weather_3.arpa ]; then
    echo "Training trigram weather lm..."
    ngram-count -order 3 -kndiscount -interpolate \
        -text $LM_INPUT/weather_wiki_simple_lm_input.txt \
        -vocab $LM_INPUT/weather_wiki_simple_vocab.txt \
        -lm $LM/domain/weather_3.arpa &
else
    echo "Skipping: Training trigram weather lm..."
fi

if [ ! -f $LM/domain/accidents_3.arpa ]; then
    echo "Training trigram accidents lm..."
    ngram-count -order 3 -kndiscount -interpolate \
        -text $LM_INPUT/accidents_wiki_simple_lm_input.txt \
        -vocab $LM_INPUT/accidents_wiki_simple_vocab.txt \
        -lm $LM/domain/accidents_3.arpa &
else
    echo "Skipping: Training trigram accidents lm..."
fi

if [ ! -f $LM/domain/terrshoot_3.arpa ]; then
    echo "Training trigram terrshoot lm..."
    ngram-count -order 3 -kndiscount -interpolate \
        -text $LM_INPUT/terrshoot_wiki_simple_lm_input.txt \
        -vocab $LM_INPUT/terrshoot_wiki_simple_vocab.txt \
        -lm $LM/domain/terrshoot_3.arpa &
else
    echo "Skipping: Training trigram terrshoot lm..."
fi

if [ ! -f $LM/domain/social_unrest_3.arpa ]; then
    echo "Training trigram social_unrest lm..."
    ngram-count -order 3 -kndiscount -interpolate \
        -text $LM_INPUT/social_unrest_wiki_simple_lm_input.txt \
        -vocab $LM_INPUT/social_unrest_wiki_simple_vocab.txt \
        -lm $LM/domain/social_unrest_3.arpa &
else
    echo "Skipping: Training trigram social_unrest lm..."
fi

wait
