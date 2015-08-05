#~/openmpi/bin/mpirun -n 10 python -u job_manager.py --cmd start --n-procs 10 \
#    --event-ids 1 2 3 4 5 6 8 9 10 \
#    --resource-paths "cuttsum.pipeline.SentenceFeaturesResource" \
#    --config job-configs.ini --service-config service-config.ini

#~/openmpi/bin/mpirun -n 40 python -u job_manager.py --cmd start --n-procs 40 \
#    --event-ids 1 2 3 4 5 6 8 9 10 \
#    --resource-paths "cuttsum.classifiers.NuggetClassifier" \
#    --config job-configs.ini --service-config service-config.ini

#~/openmpi/bin/mpirun -n 10 python -u job_manager.py --cmd start --n-procs 10 \
#    --event-ids 1 2 3 4 5 6 8 9 10  \
#    --resource-paths "cuttsum.pipeline.InputStreamResource" \
#    --config job-configs.ini --service-config service-config.ini

~/openmpi/bin/mpirun -n 10 python -u job_manager.py --cmd start --n-procs 10 \
    --event-ids 1 2 3 4 5 6 8 9 10 \
    --resource-paths "cuttsum.classifiers.NuggetRegressor" \
    --config job-configs.ini --service-config service-config.ini

