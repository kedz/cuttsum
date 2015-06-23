~/openmpi/bin/mpirun -n 5 python -u job_manager.py --cmd start --n-procs 5 \
    --event-ids 1 --resource-paths "cuttsum.pipeline.SentenceFeaturesResource" \
    --config job-configs.ini --service-config service-config.ini
#~/openmpi/bin/mpirun -n 5 python -u job_manager.py --cmd start --n-procs 5 --event-ids 1 2 3 4 5 8 9 10 --resource-paths "cuttsum.pipeline.InputStreamResource" --config job-configs.ini
