import multiprocessing


class MultiProcessWorker(object):

    def __unicode__(self):
        return unicode(self.__class__.__name__)

    def __str__(self):
        return self.__class__.__name__
 

    def do_work(self, worker, jobs, n_procs,
                progress_bar, result_handler=None, **kwargs):
        from .misc import ProgressBar
        max_jobs = len(jobs)
        manager = multiprocessing.Manager()
        job_queue = manager.Queue()
        result_queue = manager.Queue()

        for job in jobs:
            job_queue.put(job)

        pool = []        
        for i in xrange(n_procs):
            p = multiprocessing.Process(
                target=worker, args=(job_queue, result_queue), kwargs=kwargs)
            p.start()
            pool.append(p)            

            pb = ProgressBar(max_jobs)
        try:
            for n_job in xrange(max_jobs):
                result = result_queue.get(block=True)
                if result_handler is not None:
                    result_handler(result)
                if progress_bar is True:
                    pb.update()

            for p in pool:
                p.join()

        except KeyboardInterrupt:
            pb.clear()
            print "Completing current jobs and shutting down!"
            while not job_queue.empty():
                job_queue.get()
            for p in pool:
                p.join()
            sys.exit()
