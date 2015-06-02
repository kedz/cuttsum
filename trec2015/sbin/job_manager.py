from mpi4py import MPI
from cuttsum.misc import enum

tags = enum("READY", "DONE", "STOP", "ADD_JOB")

def start_manager():

    info = MPI.INFO_NULL
    port = MPI.Open_port(info)
    service = "job manager"
    MPI.Publish_name(service, info, port)
    print "I am entering loop"
    jobs = []
    while True:
        for job in jobs:
            print job
        print "I am accepting communication"
        comm = MPI.COMM_WORLD.Accept(port, info, 0)
        status = MPI.Status() # get MPI status object
        print "I am ready"
        data = comm.recv(
            source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        print "STATUS", tag, "SOURCE", source
        if tag == tags.ADD_JOB:
            jobs.extend(data)
            comm.Disconnect()
        if tag == tags.STOP:
            print "Stopping"
            comm.Disconnect()
            break

    MPI.Unpublish_name(service, info, port)

    print('closing port...')
    MPI.Close_port(port)


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
        
        exit()

        info = MPI.INFO_NULL
        port = MPI.Open_port(info)
        service = "job manager"

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
        start_manager(args.n_args)
    elif args.cmd == u"add-jobs":
        add_jobs(args.event_ids, args.resource_paths, None, **{}) #, ompi_server_file, **kwargs):
    elif args.cmd == u"stop":
        stop_manager()

    
