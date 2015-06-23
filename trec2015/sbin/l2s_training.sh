
#python vwlearner.py --learner SelectLexNextLex --training-event-ids 1 --test-event-ids 1 --report-dir "/home/t-chkedz/test-output"  \
#    --num-iters 10

#exit

python vwlearner.py --learner SelectLexNextOracle --training-event-ids 1 2 3 4 5 8 9 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-10/lexical-features-next-oracle/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextOracle --training-event-ids 1 2 3 4 5 8 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-9/lexical-features-next-oracle/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextOracle --training-event-ids 1 2 3 4 5 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-8/lexical-features-next-oracle/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextOracle --training-event-ids 1 2 3 4 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-5/lexical-features-next-oracle/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextOracle --training-event-ids 1 2 3 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-4/lexical-features-next-oracle/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextOracle --training-event-ids 1 2 4 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-3/lexical-features-next-oracle/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextOracle --training-event-ids 1 3 4 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-2/lexical-features-next-oracle/"  \
    --num-iters 10 & 

python vwlearner.py --learner SelectLexNextOracle --training-event-ids 2 3 4 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-1/lexical-features-next-oracle/"  \
    --num-iters 10 &

wait

python vwlearner.py --learner SelectLexNextLex --training-event-ids 1 2 3 4 5 8 9 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-10/lexical-features-next-lexical/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextLex --training-event-ids 1 2 3 4 5 8 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-9/lexical-features-next-lexical/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextLex --training-event-ids 1 2 3 4 5 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-8/lexical-features-next-lexical/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextLex --training-event-ids 1 2 3 4 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-5/lexical-features-next-lexical/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextLex --training-event-ids 1 2 3 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-4/lexical-features-next-lexical/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextLex --training-event-ids 1 2 4 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-3/lexical-features-next-lexical/"  \
    --num-iters 10 &

python vwlearner.py --learner SelectLexNextLex --training-event-ids 1 3 4 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-2/lexical-features-next-lexical/"  \
    --num-iters 10 & 

python vwlearner.py --learner SelectLexNextLex --training-event-ids 2 3 4 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/06-22/test-1/lexical-features-next-lexical/"  \
    --num-iters 10 &

wait



exit
python vwlearner.py --learner LexicalFeaturesNextOracle --training-event-ids 1 2 3 4 5 8 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/test-9/lexical-features-next-oracle/"  \
    --num-iters 10

python vwlearner.py --learner LexicalFeaturesNextOracle --training-event-ids 1 2 3 4 5 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/test-8/lexical-features-next-oracle/"  \
    --num-iters 10

python vwlearner.py --learner LexicalFeaturesNextOracle --training-event-ids 1 2 3 4 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/test-5/lexical-features-next-oracle/"  \
    --num-iters 10

python vwlearner.py --learner LexicalFeaturesNextOracle --training-event-ids 1 2 3 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/test-4/lexical-features-next-oracle/"  \
    --num-iters 10

python vwlearner.py --learner LexicalFeaturesNextOracle --training-event-ids 1 2 4 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/test-3/lexical-features-next-oracle/"  \
    --num-iters 10

python vwlearner.py --learner LexicalFeaturesNextOracle --training-event-ids 1 3 4 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/test-2/lexical-features-next-oracle/"  \
    --num-iters 10

python vwlearner.py --learner LexicalFeaturesNextOracle --training-event-ids 2 3 4 5 8 9 10 --test-event-ids 1 2 3 4 5 8 9 10 --report-dir "/home/t-chkedz/training-reports/test-1/lexical-features-next-oracle/"  \
    --num-iters 10

#python vwlearner.py --learner LessPerfectOracle --training-event-ids 1  --report-dir "/home/t-chkedz/training-reports/less-perfect-oracle/"  \
#    --num-iters 1 
#python vwlearner.py --learner PerfectOracle --training-event-ids 1  --report-dir "/home/t-chkedz/training-reports/perfect-oracle/"  \
#    --num-iters 1 
