~/openmpi/bin/mpirun -n 10 python -u job_manager.py --cmd start --n-procs 10 --event-ids 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46  \
    --resource-paths "cuttsum.pipeline.ArticlesResource" --config job-configs.ini \
    --service-config service-config.ini

~/openmpi/bin/mpirun -n 6 python -u job_manager.py --cmd start --n-procs 6 --event-ids 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 --resource-paths "cuttsum.pipeline.DedupedArticlesResource" --config job-configs.ini \
    --service-config service-config.ini

