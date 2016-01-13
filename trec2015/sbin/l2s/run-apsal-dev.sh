SIMS=".1 .15"
SIMS="$SIMS .2 .25"
SIMS="$SIMS .3 .35"
SIMS="$SIMS .4 .45"
SIMS="$SIMS .5 .55"
SIMS="$SIMS .6 .65"
SIMS="$SIMS .7 .75"
SIMS="$SIMS .8 .85"
SIMS="$SIMS .9"

BUCKETS="1 2 4 8 12 16 20 24"
OFF="3 3.5 4 4.5 5 5.5 6 10 15 20"

#parallel echo  ::: $SIMS ::: $BUCKETS ::: $OFF
parallel --results apsal-dev-logs -j 3 python -u \
    apsal-dev.py --output-dir apsal-dev \
    --sim-cutoff {1} --bucket-size {2} --pref-offset {3} \
    ::: $SIMS ::: $BUCKETS ::: $OFF
    
                     
