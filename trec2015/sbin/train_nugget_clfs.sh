
~/openmpi/bin/mpirun --ompi-server file:/home/t-chkedz/projects2015/trec/ompi-server.txt -n 6 python -u job_manager.py --cmd start --n-procs 6 --event-ids 1 2 3 4 5 6 --resource-paths "cuttsum.classifiers.NuggetClassifier"
