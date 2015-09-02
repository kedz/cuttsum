

#stemmed, lemmatized, or null
#stopped or not stopped
#most_mentioned ne's

for page in "terrorism" "natural-disasters" "social-unrest" "accidents"; do
    for norm in stem lemma none; do
        for is_stopped in "--stop" " "; do
            for ne in "--ne" " "; do
            echo "python lm-preprocess.py --input-dir wp-pages/$page --output wp-lm-preproc/$page --norm $norm $is_stopped $ne" 
            python lm-preprocessor.py --input-dir wp-pages/$page --output wp-lm-preproc/$page --norm $norm $is_stopped $ne
            done
        done
    done
done

