
#~/openmpi/bin/mpirun --ompi-server file:/home/t-chkedz/projects2015/trec/ompi-server.txt -n 8 python -u job_manager.py --cmd start --n-procs 8 --event-ids 1 --resource-paths "cuttsum.trecdata.UrlListResource"
#~/openmpi/bin/mpirun --ompi-server file:/home/t-chkedz/projects2015/trec/ompi-server.txt -n 8 python -u job_manager.py --cmd start --n-procs 8 --event-ids 1 --resource-paths "cuttsum.trecdata.SCChunkResource"
~/.openmpi/bin/mpirun -n 3 python -u ../job_manager.py --cmd start --n-procs 3 --event-ids 1 --resource-paths "cuttsum.pipeline.ArticlesResource" 
