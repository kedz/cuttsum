source $TREC_VENV/bin/activate
export DATA=$TREC/data2

echo "COMPUTE NUGGET SIMILARITIES"
echo "==========================="
bash $CUTTSUM/model-training-scripts/bash/compute-nugget-similarities.sh
echo

echo "COMPUTE SENTENCE FEATURES"
echo "============================"
#bash $CUTTSUM/model-training-scripts/bash/compute-sentence-features.sh
echo

echo "FITTING SALIENCE REGRESSION MODELS"
echo "=================================="
#bash $CUTTSUM/model-training-scripts/bash/fit-salience-regressions.sh
echo

echo "EXTRACTING SENTENCES AND BOW VECTORS"
echo "===================================="
#bash $CUTTSUM/model-training-scripts/bash/compute-sentence-bow.sh
echo



echo "Finished training models."
