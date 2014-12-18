
if [ ! -d $DATA/summarizer-output/models/2012_Buenos_Aires_Rail_Disaster ]; then
    echo "Building model summaries for 2012_Buenos_Aires_Rail_Disaster"
    python -u $CUTTSUM/summarize-scripts/python/build_model_summaries.py \
        -t "2012 Buenos Aires Rail Disaster" \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -o $DATA/summarizer-output/models/2012_Buenos_Aires_Rail_Disaster &
else
    echo "Skipping: Building model summaries for 2012_Buenos_Aires_Rail_Disaster"
fi

if [ ! -d $DATA/summarizer-output/models/2012_Pakistan_garment_factory_fires ]; then
    echo "Building model summaries for 2012_Pakistan_garment_factory_fires"
    python -u $CUTTSUM/summarize-scripts/python/build_model_summaries.py \
        -t "2012 Pakistan garment factory fires" \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -o $DATA/summarizer-output/models/2012_Pakistan_garment_factory_fires &
else
    echo "Skipping: Building model summaries for 2012_Pakistan_garment_factory_fires"
fi

if [ ! -d $DATA/summarizer-output/models/2012_Aurora_shooting ]; then
    echo "Building model summaries for 2012_Aurora_shooting"
    python -u $CUTTSUM/summarize-scripts/python/build_model_summaries.py \
        -t "2012 Aurora shooting" \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -o $DATA/summarizer-output/models/2012_Aurora_shooting &
else
    echo "Skipping: Building model summaries for 2012_Aurora_shooting"
fi

if [ ! -d $DATA/summarizer-output/models/Wisconsin_Sikh_temple_shooting ]; then
    echo "Building model summaries for Wisconsin_Sikh_temple_shooting"
    python -u $CUTTSUM/summarize-scripts/python/build_model_summaries.py \
        -t "Wisconsin Sikh temple shooting" \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -o $DATA/summarizer-output/models/Wisconsin_Sikh_temple_shooting &
else
    echo "Skipping: Building model summaries for Wisconsin_Sikh_temple_shooting"
fi

if [ ! -d $DATA/summarizer-output/models/Hurricane_Isaac_2012 ]; then
    echo "Building model summaries for Hurricane_Isaac_2012"
    python -u $CUTTSUM/summarize-scripts/python/build_model_summaries.py \
        -t "Hurricane Isaac (2012)" \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -o $DATA/summarizer-output/models/Hurricane_Isaac_2012 &
else
    echo "Skipping: Building model summaries for Hurricane_Isaac_2012"
fi

if [ ! -d $DATA/summarizer-output/models/Hurricane_Sandy ]; then
    echo "Building model summaries for Hurricane_Sandy"
    python -u $CUTTSUM/summarize-scripts/python/build_model_summaries.py \
        -t "Hurricane Sandy" \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -o $DATA/summarizer-output/models/Hurricane_Sandy &
else
    echo "Skipping: Building model summaries for Hurricane_Sandy"
fi

if [ ! -d $DATA/summarizer-output/models/June_2012_North_American_derecho ]; then
    echo "Building model summaries for June_2012_North_American_derecho"
    python -u $CUTTSUM/summarize-scripts/python/build_model_summaries.py \
        -t "June 2012 North American derecho" \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -o $DATA/summarizer-output/models/June_2012_North_American_derecho &
else
    echo "Skipping: Building model summaries for June_2012_North_American_derecho"
fi

if [ ! -d $DATA/summarizer-output/models/Typhoon_Bopha ]; then
    echo "Building model summaries for Typhoon_Bopha"
    python -u $CUTTSUM/summarize-scripts/python/build_model_summaries.py \
        -t "Typhoon Bopha" \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -o $DATA/summarizer-output/models/Typhoon_Bopha &
else
    echo "Skipping: Building model summaries for Typhoon_Bopha"
fi

if [ ! -d $DATA/summarizer-output/models/2012_Guatemala_earthquake ]; then
    echo "Building model summaries for 2012_Guatemala_earthquake"
    python -u $CUTTSUM/summarize-scripts/python/build_model_summaries.py \
        -t "2012 Guatemala earthquake" \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -o $DATA/summarizer-output/models/2012_Guatemala_earthquake &
else
    echo "Skipping: Building model summaries for 2012_Guatemala_earthquake"
fi

if [ ! -d $DATA/summarizer-output/models/2012_Tel_Aviv_bus_bombing ]; then
    echo "Building model summaries for 2012_Tel_Aviv_bus_bombing"
    python -u $CUTTSUM/summarize-scripts/python/build_model_summaries.py \
        -t "2012 Tel Aviv bus bombing" \
        -e $DATA/events/test-events-2013.xml \
        -n $DATA/nuggets/test-nuggets-2013.xml \
        -o $DATA/summarizer-output/models/2012_Tel_Aviv_bus_bombing &
else
    echo "Skipping: Building model summaries for 2012_Tel_Aviv_bus_bombing"
fi

wait
