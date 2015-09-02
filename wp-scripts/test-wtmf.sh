
OMP_THREAD_LIMIT=1 python test-wtmf.py --input wp-wtmf-models/accidents --norm stem --stop --lam 10.
exit

for page in "accidents"; do
    for norm in stem lemma none; do
        for is_stopped in "--stop" " "; do
            for ne in "--ne" " "; do
                for lam in ".1" "1." "10." "20."; do

OMP_THREAD_LIMIT=1 python test-wtmf.py --input wp-wtmf-models/$page --norm $norm $is_stopped $ne --lam $lam 
exit
                done
            done
        done
    done
done


