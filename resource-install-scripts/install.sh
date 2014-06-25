source $TREC_VENV/bin/activate

python -u $CUTTSUM/resource-install-scripts/gigaword2txt.py -g $GIGAWORD_DATA -of $LM_INPUT/gigaword_lm_input -r '(nyt|apw)_eng_(2009|2008|2007|2006|2005)'

