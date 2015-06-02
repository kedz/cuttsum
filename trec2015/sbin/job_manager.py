from mpi4py import MPI
from cuttsum.misc import enum

tags = enum("READY", "DONE", "STOP", "ADD_JOB", "WORKER_START", "WORKER_STOP")

def start_manager(jobs):

    comm = MPI.COMM_WORLD #.Accept(port, info, 0)
    status = MPI.Status() # get MPI status object
    #info = MPI.INFO_NULL
    #port = MPI.Open_port(info)
    #service = "job manager"
    #MPI.Publish_name(service, info, port)
    print "I am entering loop"
    idle_workers = []
    n_workers = comm.size - 1
    while n_workers > 0:

        #if len(jobs) == 0:
            #for source in idle_workers:
            #    comm.send(None, dest=dest, tag=tags.WORKER_STOP)
            #break

        for job in jobs[0:5]:
            print job
        if len(jobs) > 5:
            print "..."
        print "Remaining {:5.2f}".format(len(jobs))
        #print "I am accepting communication"
        print "I am ready"
        data = comm.recv(
            source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        print "STATUS", tag, "SOURCE", source
        if tag == tags.ADD_JOB:
            jobs.extend(data)
            #comm.Disconnect()
        if tag == tags.STOP:
            print "Stopping"
            #comm.Disconnect()
            break
        if tag == tags.READY:
            if len(jobs) > 0:
                print "sending job", jobs[0], "to worker-{}".format(source)
                comm.send(jobs.pop(0), dest=source, tag=tags.WORKER_START)
            else:
                #print "Adding idle worker-{}".format(source)
                #idle_workers.append(source)
                
                comm.send(None, dest=source, tag=tags.WORKER_STOP)
                n_workers -= 1
                if n_workers == 0:
                    break

    #MPI.Unpublish_name(service, info, port)
    #print('closing port...')
    #MPI.Close_port(port)


def stop_manager():
    
    rank = MPI.COMM_WORLD.Get_rank()
    print rank
    if rank == 0:
        info = MPI.INFO_NULL
        service = "job manager"
        port = MPI.Lookup_name(service, info)
        comm = MPI.COMM_WORLD.Connect(port, info, rank)
        comm.send([], dest=0, tag=tags.STOP)
        comm.Disconnect() 


def make_jobs(event_ids, resource_paths):
    import cuttsum.events
    import cuttsum.corpora
    events = [event for event in cuttsum.events.get_events()
              if event.query_num in set(event_ids)]
    jobs = [] 
    resources = []    
    for resource_path in resource_paths:
        mods = resource_path.split(".")
        class_name = mods[-1]
        package_path = ".".join(mods[:-1])
        mod = __import__(package_path, fromlist=[class_name])
        clazz = getattr(mod, class_name)
        resource = clazz() 
        resources.append(resource)
    
        for event in events:
            if event.query_id.startswith("TS13"):        
                corpus = cuttsum.corpora.EnglishAndUnknown2013()
            elif event.query_id.startswith("TS14"):
                corpus = cuttsum.corpora.SerifOnly2014()
            else:
                raise Exception("Bad query id: {}".format(event.query_id)) 
            for resource in resources:
                for unit in resource.get_job_units(event, corpus):
                    jobs.append((event, corpus, resource, unit))
        
    return jobs


def add_jobs(event_ids, resource_paths, ompi_server_file, **kwargs):

    import cuttsum.events
    import cuttsum.corpora
    events = [event for event in cuttsum.events.get_events()
              if event.query_num in set(event_ids)]
    
    resources = []    
    for resource_path in resource_paths:
        mods = resource_path.split(".")
        class_name = mods[-1]
        package_path = ".".join(mods[:-1])
        mod = __import__(package_path, fromlist=[class_name])
        clazz = getattr(mod, class_name)
        resource = clazz() 
        resources.append(resource)
    
    rank = MPI.COMM_WORLD.Get_rank()
    
    if rank == 0:
        info = MPI.INFO_NULL
        service = "job manager"
        port = MPI.Lookup_name(service, info)
        comm = MPI.COMM_WORLD.Connect(port, info, rank)
        
        n_jobs = 0
        n_units = 0
        jobs = []
        for event in events:
            if event.query_id.startswith("TS13"):        
                corpus = cuttsum.corpora.EnglishAndUnknown2013()
            elif event.query_id.startswith("TS14"):
                corpus = cuttsum.corpora.SerifOnly2014()
            else:
                raise Exception("Bad query id: {}".format(event.query_id)) 
            for resource in resources:
                print "Adding job", event, corpus, resource
                n_jobs += 1

                for i in xrange(5):
                    n_units += 1
                    jobs.append((event, corpus, resource, i))
        
        print "Added", n_jobs, "jobs comprising", n_units, "units of work."
        comm.send(jobs, dest=0, tag=tags.ADD_JOB)
        comm.Disconnect()

def start_worker():
    
    rank = MPI.COMM_WORLD.Get_rank()
    status = MPI.Status() # get MPI status object
    #info = MPI.INFO_NULL
    #service = "job manager"
    #port = MPI.Lookup_name(service, info)
    import time

    while True:
        #info = MPI.INFO_NULL
        #service = "job manager"
        #comm = MPI.COMM_WORLD.Connect(port, info, 0)
        comm.send(None, dest=0, tag=tags.READY)
        data = comm.recv(
            source=0, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        #comm.Disconnect()        
        if tag == tags.WORKER_START:
            event, corpus, resource, unit = data
            print "worker-{} {} {} {} {}".format(rank, event.fs_name(), corpus.fs_name(), resource, unit) 
            resource.do_job_unit(event, corpus, unit)
        if tag == tags.WORKER_STOP:
            break

    print "worker-{} shutting down!".format(rank)

if __name__ == u"__main__":
      
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(u"--cmd", type=unicode, choices=[
        u"start", u"add-jobs", u"stop"],
                        help=u"job manager command.")
    parser.add_argument(u"--event-ids", type=int, nargs=u"+",
                        help=u"event ids to select.")
    parser.add_argument(u"--resource-paths", type=unicode, nargs=u"+",
                        help=u"resources to add to job queue.",
                        choices=[ 
        "cuttsum.trecdata.UrlListResource", 
        "cuttsum.trecdata.SCChunkResource"])
    parser.add_argument(u"--n-procs", type=int, default=1,
                        help="number of processes to add")

    args = parser.parse_args()

    if args.cmd == "start":

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
        if rank == 0:
            print "starting manager!"

            jobs = make_jobs(args.event_ids, args.resource_paths)
            start_manager(jobs)
        elif rank < args.n_procs:
            #import time 
            #time.sleep(2 * rank)
            print "starting worker"
            start_worker()
    elif args.cmd == u"add-jobs":
        add_jobs(args.event_ids, args.resource_paths, None, **{}) #, ompi_server_file, **kwargs):
    elif args.cmd == u"stop":
        stop_manager()

    
