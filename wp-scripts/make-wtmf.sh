
for page in "terrorism"; do
    for norm in stem lemma none; do
        for is_stopped in "--stop" " "; do
            for ne in "--ne" " "; do
                for lam in ".1" "1." "10." "20."; do

echo "python make-wtmf-models.py --input wp-lm-preproc/$page --norm $norm $is_stopped $ne --lam $lam"             
OMP_THREAD_LIMIT=1 python make-wtmf-models.py --input wp-lm-preproc/$page --output wp-wtmf-models/$page --norm $norm $is_stopped $ne --lam $lam &

                done
            done
            wait
        done
    done
done

