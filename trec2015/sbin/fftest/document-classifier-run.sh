#~/openmpi/bin/mpirun -n 1 \
    python -u document-classifier-run.py --features averaged-basic \
    --training-event-ids 1 2 3 4 5 6 8 9 10  --test-event-ids 1 \
    --report-dir /home/t-chkedz/test-run \
    --num-iters 10 \
    --sample-size 90 
# \
#    averaged-basic markov-next  doc-size averaged-lm \
exit
#~/openmpi/bin/mpirun -n 1 \
    python -u document-classifier-run.py --features markov-next update-input-lex  averaged-basic \
    --training-event-ids 1 3 4 5 6 7 8 9 10  --test-event-ids 2 \
    --report-dir /home/t-chkedz/test-run \
    --num-iters 10 \
    --sample-size 90 &>log2 &

#~/openmpi/bin/mpirun -n 1 \
    python -u document-classifier-run.py --features markov-next update-input-lex  averaged-basic \
    --training-event-ids 1 2 4 5 6 7 8 9 10  --test-event-ids 3 \
    --report-dir /home/t-chkedz/test-run \
    --num-iters 10 \
    --sample-size 90 &>log3 &

#~/openmpi/bin/mpirun -n 1 \
    python -u document-classifier-run.py --features markov-next update-input-lex  averaged-basic \
    --training-event-ids 1 2 3 5 6 7 8 9 10  --test-event-ids 4 \
    --report-dir /home/t-chkedz/test-run \
    --num-iters 10 \
    --sample-size 90 &>log4 &

#~/openmpi/bin/mpirun -n 1 \
    python -u document-classifier-run.py --features markov-next update-input-lex  averaged-basic \
    --training-event-ids 1 2 3 4  6 7 8 9 10  --test-event-ids 5 \
    --report-dir /home/t-chkedz/test-run \
    --num-iters 10 \
    --sample-size 90 &>log5 &

#~/openmpi/bin/mpirun -n 1 \
    python -u document-classifier-run.py --features markov-next update-input-lex  averaged-basic \
    --training-event-ids 1 2 3 4 5 7 8 9 10  --test-event-ids 6 \
    --report-dir /home/t-chkedz/test-run \
    --num-iters 10 \
    --sample-size 90 &>log6 &

#~/openmpi/bin/mpirun -n 1 \
    python -u document-classifier-run.py --features markov-next update-input-lex  averaged-basic \
    --training-event-ids 1 2 3 4 5 6 7 9 10  --test-event-ids 8 \
    --report-dir /home/t-chkedz/test-run \
    --num-iters 10 \
    --sample-size 90 &>log8 &

wait

#~/openmpi/bin/mpirun -n 1 \
    python -u document-classifier-run.py --features markov-next update-input-lex  averaged-basic \
    --training-event-ids 1 2 3 4 5 6 7 8 10  --test-event-ids 9 \
    --report-dir /home/t-chkedz/test-run \
    --num-iters 10 \
    --sample-size 90 &>log9 &

#~/openmpi/bin/mpirun -n 1 \
    python -u document-classifier-run.py --features markov-next update-input-lex  averaged-basic \
    --training-event-ids 1 2 3 4 5 6 7 8 9   --test-event-ids 10 \
    --report-dir /home/t-chkedz/test-run \
    --num-iters 10 \
    --sample-size 90 &>log10 &






