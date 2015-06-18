#~/openmpi/bin/mpirun -n 2 python -u job_manager.py --cmd start --n-procs 2 --event-ids 1  --resource-paths "cuttsum.pipeline.SentenceFeaturesResource" --config job-configs.ini
~/openmpi/bin/mpirun -n 3 python -u job_manager.py --cmd start --n-procs 3 --event-ids 1 2 3 --resource-paths "cuttsum.pipeline.InputStreamResource" --config job-configs.ini
