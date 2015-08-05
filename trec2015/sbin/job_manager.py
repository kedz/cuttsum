from mpi4py import MPI
from cuttsum.misc import enum
from datetime import datetime


tags = enum("READY", "DONE", "STOP", "ADD_JOB", "WORKER_START", "WORKER_STOP")

def start_service(service, service_configs):
    if service == "corenlp":
        cnlp_config = service_configs.get("corenlp", {})
        mem = cnlp_config.get("mem", "8G")
        threads = int(cnlp_config.get("threads", 4))
        max_message_len = int(cnlp_config.get("max_message_len", 524288))
        port = int(cnlp_config.get("port", 9999))
        import corenlp as cnlp
        cnlp.server.start(
            port=port,
            mem=mem, threads=threads, max_message_len=max_message_len,
            annotators=["tokenize", "ssplit", "pos", "lemma", "ner", "parse"],
            corenlp_props={
                "pos.maxlen": "150",
                "ssplit.eolonly": "true"})
    elif service.endswith("-lm"):
        lm_config = service_configs[service]
        path = lm_config["path"]    
        port = int(lm_config["port"])
	order = int(lm_config.get("order", 3))
        print "starting", service, " on port", port
        import cuttsum.srilm as srilm
        pid = srilm.start_lm(path, order, port)
        lm_config["pid"] = pid
        print "lm is started with pid", pid

def stop_service(service, service_configs):
    if service == "corenlp":
        cnlp_config = service_configs.get("corenlp", {})
        port = int(cnlp_config.get("port", 9999))
        print "KILLING CORENLP"
        import corenlp as cnlp
        cnlp.server.stop(port=port)
    if service.endswith("-lm"):
        lm_config = service_configs[service]
        pid = lm_config["pid"]
        import signal
        import os
        print "killing", service, "with pid", pid
        os.kill(pid, signal.SIGKILL)

def start_manager(jobs, service_configs):

    comm = MPI.COMM_WORLD #.Accept(port, info, 0)
    status = MPI.Status() # get MPI status object
    #info = MPI.INFO_NULL
    #port = MPI.Open_port(info)
    #service = "job manager"
    #MPI.Publish_name(service, info, port)

    services = {}

    idle_workers = []
    n_workers = comm.size - 1
    while n_workers > 0:

        #if len(jobs) == 0:
            #for source in idle_workers:
            #    comm.send(None, dest=dest, tag=tags.WORKER_STOP)
            #break

        for job in jobs[0:5]:
            event, corpus, res, unit, job_name, kwargs = job
            print job_name, event.fs_name(), corpus.fs_name(), res, unit
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
                #print "sending job", jobs[0], "to worker-{}".format(source)
                for req in data:
                    services[req] -= 1
 
                job = jobs.pop(0)
                event, corpus, resource, unit, job_name, kwargs = job
                requires = resource.requires_services(
                    event, corpus, **kwargs)
                for req in requires:
                    num_running = services.get(req, 0)
                    if num_running == 0:
                        start_service(req, service_configs)
                    services[req] = num_running + 1

                comm.send(job, dest=source, tag=tags.WORKER_START)
                
                for req, num_users in services.items():
                    if num_users == 0:
                        stop_service(req, service_configs)
            else:

                for req, num_users in services.items():
                    print req, num_users
                    if num_users == 0:
                        stop_service(req, service_configs)
                #print "Adding idle worker-{}".format(source)
                #idle_workers.append(source)
                comm.send(None, dest=source, tag=tags.WORKER_STOP)
                n_workers -= 1
                if n_workers == 0:
                    break
    for service in services.keys():
        stop_service(service, service_configs)
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


def make_job_configurations(config_path):

    from collections import defaultdict
    type_conv = defaultdict(lambda: lambda val, opt, sec, path: val) 
    def bool_checker(val, opt, section, path):
        if val in ["True", "true"]:
            return True
        elif val in ["False", "false"]:
            return False
        else:
            raise Exception(
                ("In section {} bad config option {} with value {} " + \
                 "in config file {}").format(section, opt, val, path))

    def int_checker(val, opt, section, path):
        try:
            return int(val)
        except ValueError:
            raise Exception(
                ("In section {} bad config option {} with value {} " + \
                 "in config file {}").format(section, opt, val, path))


    type_conv["soft_match"] = bool_checker
    type_conv["overwrite"] = bool_checker
    type_conv["preroll"] = int_checker

    job_configurations = defaultdict(list)
    if config_path is not None:
        from ConfigParser import ConfigParser
        cp = ConfigParser()
        cp.read(config_path) 
        for section in cp.sections():
            if cp.has_option(section, "class"):
                clazz = None
                params = {}    
                for key, val in cp.items(section):
                    if key == "class":
                        clazz = val
                    else:  
                        params[key] = type_conv[key](
                            val, key, section, config_path)                  
                job_configurations[clazz].append((section, params))
    return job_configurations  

def make_jobs(event_ids, resource_paths, config_path, service_configs):

    configs = make_job_configurations(config_path)
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
        
        res_configs = configs[resource_path]
        resources.append((resource, res_configs))
    
        for event in events:
            corpus = cuttsum.corpora.get_raw_corpus(event)
            for resource, jobs_settings in resources:
                if len(jobs_settings) == 0:                
                    for unit in resource.get_job_units(event, corpus):
                        jobs.append(
                            (event, corpus, resource, unit, "default", 
                             {"service-configs": service_configs}))
                else:
                    for job_name, job_settings in jobs_settings:
                        job_settings["service-configs"] = service_configs
                        for unit in resource.get_job_units(
                                event, corpus, **job_settings):
                            jobs.append(
                                (event, corpus, resource, unit, 
                                 job_name, job_settings))
                           
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

    done_with_services = []
    while True:
        #info = MPI.INFO_NULL
        #service = "job manager"
        #comm = MPI.COMM_WORLD.Connect(port, info, 0)
        comm.send(done_with_services, dest=0, tag=tags.READY)
        data = comm.recv(
            source=0, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        #comm.Disconnect()        
        if tag == tags.WORKER_START:
            event, corpus, resource, unit, job_name, kwargs = data
            print "worker-{} {} {} {} {} {}".format(
                rank, job_name, event.fs_name(), corpus.fs_name(), resource, unit) 
            resource.do_job_unit(event, corpus, unit, **kwargs)
            done_with_services = resource.requires_services(
                event, corpus, **kwargs)
        if tag == tags.WORKER_STOP:
            break

    print "worker-{} shutting down!".format(rank)

def parse_service_config(config_path):
    from ConfigParser import ConfigParser
    parser = ConfigParser()
    parser.read(config_path)
    params = {}
    for section in parser.sections():
        params[section] = {}
        for key, val in parser.items(section):
            params[section][key] = val
    return params

if __name__ == u"__main__":
    
    import sys  
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(u"--cmd", type=unicode, choices=[
        u"start", u"add-jobs", u"stop"],
                        help=u"job manager command.")
    parser.add_argument(u"--event-ids", type=int, nargs=u"+",
                        help=u"event ids to select.")
    parser.add_argument(u"--resource-paths", type=str, nargs=u"+",
                        help=u"resources to add to job queue.",
                        choices=[ 
        "cuttsum.trecdata.UrlListResource", 
        "cuttsum.trecdata.SCChunkResource",
        "cuttsum.trecdata.SCChunkValidatorResource",
        "cuttsum.pipeline.ArticlesResource",
        "cuttsum.pipeline.DedupedArticlesResource",
        "cuttsum.pipeline.SentenceFeaturesResource",
        "cuttsum.pipeline.InputStreamResource",
        "cuttsum.classifiers.NuggetClassifier",
        "cuttsum.classifiers.NuggetRegressor",
        "cuttsum.summarizers.RetrospectiveMonotoneSubmodularOracle",
        "cuttsum.summarizers.MonotoneSubmodularOracle",
        ])
    parser.add_argument(u"--n-procs", type=int, default=1,
                        help="number of processes to add")
    parser.add_argument(u"--config", type=str, default=None,
                        help=u"path to config file.")

    parser.add_argument(u"--service-config", type=str, default=None,
                        help=u"path to service config file.")

    args = parser.parse_args()
    service_configs = parse_service_config(args.service_config)

    if args.cmd == "start":

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size

        if rank == 0:

            print "Starting CUTTSUM job manager."
            jobs = make_jobs(
                args.event_ids, args.resource_paths, args.config,
                service_configs)
            msg = "{}: Running ``job_manager.py" + " ".join(sys.argv) + "''"
            start = datetime.now()
            with open("jm.log", "a") as log:
                log.write(msg.format(start) + "\n")
                log.flush()
                try:
                    start_manager(jobs, service_configs)
                finally:
                    stop = datetime.now()
                    duration = stop - start
                    msg = "{}: Finished ``job_manager.py" \
                        + " ".join(sys.argv) + "''" \
                        + " Elapsed time: {}"
                    log.write(msg.format(stop, duration) + "\n")
                    log.flush()
                exit()         
        elif rank < args.n_procs:
            start_worker()
    elif args.cmd == u"add-jobs":
        add_jobs(args.event_ids, args.resource_paths, None, **{})
    elif args.cmd == u"stop":
        stop_manager()

    
