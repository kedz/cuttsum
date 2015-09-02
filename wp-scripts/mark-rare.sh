


for page in "accidents" "natural-disasters" "terrorism" "social-unrest"; do
    for norm in stem lemma none; do
        for is_stopped in "--stop" " "; do

    echo "python mark-rare.py --input wp-lm-preproc/$page --output wp-lm-preproc-unk/$page --norm $norm $is_stopped --threshold 3"
    python mark-rare.py --input wp-lm-preproc/$page --output wp-lm-preproc-unk/$page --norm $norm $is_stopped --threshold 3

        done
    done
done

if [ ! -d "wp-lm" ]; then
    mkdir "wp-lm"
fi

echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-stem.stop.spl.unk.gz -lm wp-lm/accidents.norm-stem.stop.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-stem.stop.spl.unk.gz -lm wp-lm/accidents.norm-stem.stop.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-stem.spl.unk.gz -lm wp-lm/accidents.norm-stem.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-stem.spl.unk.gz -lm wp-lm/accidents.norm-stem.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-lemma.stop.spl.unk.gz -lm wp-lm/accidents.norm-lemma.stop.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-lemma.stop.spl.unk.gz -lm wp-lm/accidents.norm-lemma.stop.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-lemma.stop.spl.unk.gz -lm wp-lm/accidents.norm-lemma.stop.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-lemma.stop.spl.unk.gz -lm wp-lm/accidents.norm-lemma.stop.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-lemma.spl.unk.gz -lm wp-lm/accidents.norm-lemma.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-lemma.spl.unk.gz -lm wp-lm/accidents.norm-lemma.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-none.stop.spl.unk.gz -lm wp-lm/accidents.norm-none.stop.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-none.stop.spl.unk.gz -lm wp-lm/accidents.norm-none.stop.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-none.spl.unk.gz -lm wp-lm/accidents.norm-none.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/accidents.norm-none.spl.unk.gz -lm wp-lm/accidents.norm-none.3.arpa  

echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-stem.stop.spl.unk.gz -lm wp-lm/natural-disasters.norm-stem.stop.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-stem.stop.spl.unk.gz -lm wp-lm/natural-disasters.norm-stem.stop.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-stem.spl.unk.gz -lm wp-lm/natural-disasters.norm-stem.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-stem.spl.unk.gz -lm wp-lm/natural-disasters.norm-stem.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-lemma.stop.spl.unk.gz -lm wp-lm/natural-disasters.norm-lemma.stop.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-lemma.stop.spl.unk.gz -lm wp-lm/natural-disasters.norm-lemma.stop.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-lemma.spl.unk.gz -lm wp-lm/natural-disasters.norm-lemma.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-lemma.spl.unk.gz -lm wp-lm/natural-disasters.norm-lemma.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-none.stop.spl.unk.gz -lm wp-lm/natural-disasters.norm-none.stop.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-none.stop.spl.unk.gz -lm wp-lm/natural-disasters.norm-none.stop.3.arpa   
echo "ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-none.spl.unk.gz -lm wp-lm/natural-disasters.norm-none.3.arpa"
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/natural-disasters.norm-none.spl.unk.gz -lm wp-lm/natural-disasters.norm-none.3.arpa   

ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/terrorism.norm-stem.stop.spl.unk.gz -lm wp-lm/terrorism.norm-stem.stop.3.arpa   
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/terrorism.norm-stem.spl.unk.gz -lm wp-lm/terrorism.norm-stem.3.arpa   
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/terrorism.norm-lemma.stop.spl.unk.gz -lm wp-lm/terrorism.norm-lemma.stop.3.arpa   
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/terrorism.norm-lemma.spl.unk.gz -lm wp-lm/terrorism.norm-lemma.3.arpa   
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/terrorism.norm-none.stop.spl.unk.gz -lm wp-lm/terrorism.norm-none.stop.3.arpa   
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/terrorism.norm-none.spl.unk.gz -lm wp-lm/terrorism.norm-none.3.arpa   

ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/social-unrest.norm-stem.stop.spl.unk.gz -lm wp-lm/social-unrest.norm-stem.stop.3.arpa   
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/social-unrest.norm-stem.spl.unk.gz -lm wp-lm/social-unrest.norm-stem.3.arpa   
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/social-unrest.norm-lemma.stop.spl.unk.gz -lm wp-lm/social-unrest.norm-lemma.stop.3.arpa   
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/social-unrest.norm-lemma.spl.unk.gz -lm wp-lm/social-unrest.norm-lemma.3.arpa   
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/social-unrest.norm-none.stop.spl.unk.gz -lm wp-lm/social-unrest.norm-none.stop.3.arpa   
ngram-count -order 3 -kndiscount -interpolate -text wp-lm-preproc-unk/social-unrest.norm-none.spl.unk.gz -lm wp-lm/social-unrest.norm-none.3.arpa   





