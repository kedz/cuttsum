SIMS=".1 .15"
SIMS="$SIMS .2 .25"
SIMS="$SIMS .3 .35"
SIMS="$SIMS .4 .45"
SIMS="$SIMS .5 .55"
SIMS="$SIMS .6 .65"
SIMS="$SIMS .7 .75"
SIMS="$SIMS .8 .85"
SIMS="$SIMS .9 .95 1"

parallel --results l2s-fltr-dev-logs -j 3 python -u \
    l2s-filter-dev.py --output-dir l2s-fltr-dev \
    --sim-thresh {} ::: $SIMS
