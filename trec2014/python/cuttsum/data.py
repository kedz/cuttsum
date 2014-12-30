import multiprocessing
import cuttsum
import pkgutil
import inspect
import sys
from .misc import toposort

resources_ = {}
def get_resource_manager(resource_name):
    if len(resources_) == 0:
        _init_resource_manager()
    return resources_.get(resource_name, None)

def get_resource_managers():
    if len(resources_) == 0:
        _init_resource_manager()
    return resources_.values()

def get_sorted_dependencies(reverse=False):
    data = {}
    for resource in get_resource_managers():
        data[resource] = set([get_resource_manager(dep)
                              for dep in resource.dependencies()])

    groups = [group for group in toposort(data)]
    if reverse is True:
        return reversed(groups)
    else:
        return groups

def _init_resource_manager():
    for loader, module_name, is_pkg in pkgutil.walk_packages(
        cuttsum.__path__, cuttsum.__name__ + ".", onerror=None):
        if module_name == "cuttsum.corpora":
            continue
        if module_name == "cuttsum.events":
            continue
        if module_name == "cuttsum.judgements":
            continue
        module = loader.find_module(module_name).load_module(module_name)
        clsmembers = inspect.getmembers(
            sys.modules[module.__name__],
            lambda x: inspect.isclass(x) and \
                issubclass(x, cuttsum.data.Resource))  

        for name, clazz in clsmembers:
            if name == "Resource":
                continue
            resources_[name] = clazz()


class Resource(object):
    def __init__(self):
        self.deps_met_ = set()
    
    def check_unmet_dependencies(self, event, corpus, depth=0, **kwargs):
        unmet_deps = []
        for resource_name in self.dependencies():            
            
            res = get_resource_manager(resource_name)
            if (res, event, corpus) not in self.deps_met_:
                coverage = res.check_coverage(event, corpus, **kwargs)
                if coverage == 1:
                    self.deps_met_.add((res, event, corpus))
                else:
                    unmet_deps.append((depth + 1, res, coverage))
            unmet_deps.extend(res.check_unmet_dependencies(
                event, corpus, depth=depth + 1, **kwargs))
        return unmet_deps

    def get_dependencies(self, event, corpus, priority_deps=None, **kwargs):
        if priority_deps is None:
            priority_deps = self.check_unmet_dependencies(
                event, corpus, **kwargs)
        priority_deps.sort(key=lambda x: x[0], reverse=True)
        for item in priority_deps:
            priority = item[0]
            dep = item[1]
            print "Retrieving " + str(dep) + "..."
            dep.get(event, corpus, **kwargs) 

    def check_coverage(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def dependencies(self):
        raise NotImplementedError

    def __str__(self):
        return unicode(self).encode(u'utf-8')

    @classmethod
    def getdependency(self, fun):
        def wrapper(self, event, corpus, **kwargs):            
            print "I am here"
            is_met = \
                self.deps_met_.get((event.fs_name(), corpus.fs_name()), False)
            if is_met is False and kwargs.get('no_resolve', False) is False:
                for dc in self.dependencies():
                    print self, dc
                    dep = dc()
                    coverage_per = dep.check_coverage(event, corpus, **kwargs)
                    if coverage_per != 1.:
                        sys.stdout.write(
                            u"{}: Incomplete coverage (%{:0.2f}), "
                            u"retrieving...\n"
                                .format(dep, coverage_per * 100.))
                        print dep.get
                        if dep.get(event, corpus, **kwargs) is True:
                            sys.stdout.write('OK!\n')
                            sys.stdout.flush()
                        else:
                            sys.stdout.write('FAILED!\n')
                            sys.flush()
                            sys.stderr.write(
                                'Failed to retrieve necessary resource:\n')
                            sys.stderr.write('\t{} for {} / {}'.format(
                                dep, event.query_id, corpus.fs_name())) 
                            sys.stderr.flush()
                            sys.exit(1)
            self.deps_met_[(event.fs_name(), corpus.fs_name())] = True
            return fun(self, event, corpus, **kwargs)
        return wrapper

    @classmethod
    def getsuperdependency(self, fun):
        def wrapper(self, event, corpus, **kwargs):
            if corpus.is_subset() is True:
                corpus = corpus.get_superset()
            return fun(self, event, corpus, **kwargs)
        return wrapper

    def do_work(self, worker, jobs, n_procs,
                progress_bar, result_handler=None, **kwargs):
        from .misc import ProgressBar
        max_jobs = len(jobs)
        job_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

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



