parallel --results crossval-logs -j 4 python crossval.py \
    --output-dir crossval/l2.00001/{}/ \
    --iters 5 --l2 .00001 --test-event {} ::: {1..6} {8..23} {25..46}
