#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 100 --samples-per-event 10 --iters 20 --l2 .001 --output-dir "ssize.100.spe.10.iters.20.l2..001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 200 --samples-per-event 10 --iters 20 --l2 .001 --output-dir "ssize.200.spe.10.iters.20.l2..001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 300 --samples-per-event 10 --iters 20 --l2 .001 --output-dir "ssize.300.spe.10.iters.20.l2..001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 500 --samples-per-event 10 --iters 20 --l2 .001 --output-dir "ssize.500.spe.10.iters.20.l2..001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 100 --samples-per-event 100 --iters 20 --l2 .001 --output-dir "ssize.100.spe.100.iters.20.l2..001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 200 --samples-per-event 100 --iters 20 --l2 .001 --output-dir "ssize.200.spe.100.iters.20.l2..001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 300 --samples-per-event 100 --iters 20 --l2 .001 --output-dir "ssize.300.spe.100.iters.20.l2..001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 500 --samples-per-event 100 --iters 20 --l2 .001 --output-dir "ssize.500.spe.100.iters.20.l2..001"


#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 100 --samples-per-event 10 --iters 20 --l2 .0001 --output-dir "ssize.100.spe.10.iters.20.l2..0001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 200 --samples-per-event 10 --iters 20 --l2 .0001 --output-dir "ssize.200.spe.10.iters.20.l2..0001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 300 --samples-per-event 10 --iters 20 --l2 .0001 --output-dir "ssize.300.spe.10.iters.20.l2..0001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 500 --samples-per-event 10 --iters 20 --l2 .0001 --output-dir "ssize.500.spe.10.iters.20.l2..0001"




mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6  8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25   \
    --sample-size 100 --samples-per-event 5 --iters 20 --l2 .0001 --log-time --output-dir "ssize.100.spe.5.iters.20.l2..0001.df"  

mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6  8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25   \
    --sample-size 100 --samples-per-event 10 --iters 20 --l2 .0001 --log-time --output-dir "ssize.100.spe.10.iters.20.l2..0001.df"  

mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6  8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25   \
    --sample-size 100 --samples-per-event 15 --iters 20 --l2 .0001 --log-time --output-dir "ssize.100.spe.15.iters.20.l2..0001.df"  

mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6  8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25   \
    --sample-size 300 --samples-per-event 5 --iters 20 --l2 .0001 --log-time --output-dir "ssize.300.spe.5.iters.20.l2..0001.df"  

mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6  8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25   \
    --sample-size 300 --samples-per-event 10 --iters 20 --l2 .0001 --log-time --output-dir "ssize.300.spe.10.iters.20.l2..0001.df"  

mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6  8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25   \
    --sample-size 300 --samples-per-event 15 --iters 20 --l2 .0001 --log-time --output-dir "ssize.300.spe.15.iters.20.l2..0001.df"  




#mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \


#mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
#    --sample-size 100 --samples-per-event 5 --iters 5 --l2 .0001 --log-time --best-feats --abs-df --output-dir "ssize.100.spe.5.iters.5.l2..0001.df.best_feat.abs_df"  



#mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
#    --sample-size 100 --samples-per-event 5 --iters 5 --l2 .001 --log-time --output-dir "ssize.100.spe.5.iters.5.l2..001.df"  

#mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
#    --sample-size 100 --samples-per-event 5 --iters 5 --l2 .0001 --log-time --best-feats --output-dir "ssize.100.spe.5.iters.5.l2..0001.df.best_feat"  
#mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
#    --sample-size 100 --samples-per-event 5 --iters 5 --l2 .0001 --log-time --best-feats --i-only --output-dir "ssize.100.spe.5.iters.5.l2..0001.df.best_feat.i_only"  



#mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
#    --sample-size 100 --samples-per-event 5 --iters 5 --l2 .0001 --log-time --output-dir "ssize.100.spe.5.iters.5.l2..0001.df"  &

#mpirun -n 25 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
#    --sample-size 50 --samples-per-event 5 --iters 5 --l2 .0001 --output-dir "ssize.50.spe.5.iters.5.l2..0001.df" &
#wait

#wait

#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 200 --samples-per-event 100 --iters 20 --l2 .0001 --output-dir "ssize.200.spe.100.iters.20.l2..0001" &
#wait 
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 300 --samples-per-event 100 --iters 20 --l2 .0001 --output-dir "ssize.300.spe.100.iters.20.l2..0001" &
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 500 --samples-per-event 100 --iters 20 --l2 .0001 --output-dir "ssize.500.spe.100.iters.20.l2..0001" &

#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 100 --samples-per-event 10 --iters 20 --l2 .00001 --output-dir "ssize.100.spe.10.iters.20.l2..00001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 200 --samples-per-event 10 --iters 20 --l2 .00001 --output-dir "ssize.200.spe.10.iters.20.l2..00001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 300 --samples-per-event 10 --iters 20 --l2 .00001 --output-dir "ssize.300.spe.10.iters.20.l2..00001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 500 --samples-per-event 10 --iters 20 --l2 .00001 --output-dir "ssize.500.spe.10.iters.20.l2..00001"
#
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 100 --samples-per-event 100 --iters 20 --l2 .00001 --output-dir "ssize.100.spe.100.iters.20.l2..00001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 200 --samples-per-event 100 --iters 20 --l2 .00001 --output-dir "ssize.200.spe.100.iters.20.l2..00001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 300 --samples-per-event 100 --iters 20 --l2 .00001 --output-dir "ssize.300.spe.100.iters.20.l2..00001"
#mpirun -n 10 python cross-validate-l2s.py --event-ids 1 2 3 4 5 6 8 9 10 --sample-size 500 --samples-per-event 100 --iters 20 --l2 .00001 --output-dir "ssize.500.spe.100.iters.20.l2..00001"
#
