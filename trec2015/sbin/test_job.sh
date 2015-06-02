~/.openmpi/bin/mpirun --ompi-server file:ompi-server.txt -n 1 python job_manager.py --cmd add-jobs --event-ids 1 2 3 --resource-paths cuttsum.trecdata.UrlListResource
