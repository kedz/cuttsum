#~/openmpi/bin/mpirun -n 2 python -u job_manager.py --cmd start --n-procs 2 --event-ids 1 --resource-paths "cuttsum.pipeline.ArticlesResource" --config job-configs.ini
#~/openmpi/bin/mpirun -n 6 python -u job_manager.py --cmd start --n-procs 6 --event-ids 1 2 3 4 5 6 8 9 10 --resource-paths "cuttsum.pipeline.DedupedArticlesResource" --config job-configs.ini

