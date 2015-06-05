~/openmpi/bin/mpirun --ompi-server file:/home/t-chkedz/projects2015/trec/ompi-server.txt -n 8 python -u job_manager.py --cmd start --n-procs 8 --event-ids 1 2 3 4 5 6 8 9 10 --resource-paths "cuttsum.pipeline.ArticlesResource" --config job-configs.ini

