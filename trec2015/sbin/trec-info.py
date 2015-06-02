import cuttsum.events
import cuttsum.corpora
import numpy as np
import sys
from blessed import Terminal 
import readline

def print_event_timelines(term, events):
    #events.sort(key=lambda x: x.start)
    min_date = np.min([event.start for event in events])
    max_date = np.max([event.end for event in events])

    with term.location(0,0):
        print term.clear

        term_width = term.width
        col_width = int(np.ceil(.85 * term_width))
        title_width = term_width - col_width - 1

        delta = (max_date - min_date) / col_width

        t = min_date
        sys.stdout.write(" "*title_width)
        while t < max_date:
            if t.year != (t - delta).year:
                sys.stdout.write("*")
            elif t.month != (t - delta).month:
                sys.stdout.write(".")
            else:
                sys.stdout.write(" ")

            t += delta
        sys.stdout.write("\n")

        for event in events:
            sys.stdout.write(
                ("{:" + str(title_width-1) + "s}").format(
                    event.title[0:title_width-1]))
            sys.stdout.write(" ")

            t = min_date
            while t < max_date:
                if t >= event.start and t <= event.end:
                    sys.stdout.write("#")
                else:
                    sys.stdout.write(" ")

                t += delta
            sys.stdout.write("\n")
        with term.cbreak():
            term.inkey()


def home_screen(t, events):

    with t.location(0, 0): #t.height - 1):
        print t.clear
        if len(events) == 0:
            return 
        print("==" + t.bold('Active Events') + "=========================")
        #print(t.bold_red_on_bright_green('It hurts my eyes!'))
        for event in events:
            print("||  " + \
                t.green_on_black("{}) {}".format(
                    event.query_num, event.title)))
        print("========================================")

        #print(t.center('press any key to continue.'))

        #with t.cbreak():
        #    t.inkey()




def cmd_select_events(args, active_events, all_events):
    qnum = set()
    for arg in args:
        try:
            qnum.add(int(arg))
        except ValueError:
            pass

    selected = [event for event in all_events 
                if event.query_num in qnum]
    for event in selected:
        if event not in active_events:
            active_events.append(event)
    active_events.sort(key=lambda x: x.query_num)

def cmd_deselect_events(args, active_events):
    qnums = set()
    for arg in args:
        try:
            qnums.add(int(arg))
        except ValueError:
            pass
    return [event for event in active_events
            if event.query_num not in qnums]


def cmd_status(args, t, active_events):

    print t.clear
    for event in active_events:
        print "{}) {}".format(event.query_num, event.title)
        if event.query_id.startswith("TS13"):        
            corpus = cuttsum.corpora.EnglishAndUnknown2013()
        elif event.query_id.startswith("TS14"):
            corpus = cuttsum.corpora.SerifOnly2014()
        else:
            raise Exception("Bad query id: {}".format(event.query_id)) 
        for arg in args:
            print arg
            mods = arg.split(".")
            class_name = mods[-1]
            package_path = ".".join(mods[:-1])
            mod = __import__(package_path, fromlist=[class_name])
            clazz = getattr(mod, class_name)
            resource = clazz()        
            print "{} {:3.2f}%".format(resource, 100. * resource.check_coverage(event, corpus, preroll=0))
    with t.cbreak():
        t.inkey()



def cmd_run(args, t, active_events):
    args, kwargs = simple_kw_parse(args)
    print t.clear
    for event in active_events:
        print "{}) {}".format(event.query_num, event.title)
        if event.query_id.startswith("TS13"):        
            corpus = cuttsum.corpora.EnglishAndUnknown2013()
        elif event.query_id.startswith("TS14"):
            corpus = cuttsum.corpora.SerifOnly2014()
        else:
            raise Exception("Bad query id: {}".format(event.query_id)) 
        for arg in args:
            print arg
            mods = arg.split(".")
            class_name = mods[-1]
            package_path = ".".join(mods[:-1])
            mod = __import__(package_path, fromlist=[class_name])
            clazz = getattr(mod, class_name)
            resource = clazz()        
            resource.get(event, corpus, **kwargs)
#overwrite=False, n_procs=1,
#                         progress_bar=True, preroll=0)


def simple_kw_parse(args):
    reg_args = []
    kwargs = {}
    for arg in args:
        key_val = arg.split("=")
        if len(key_val) == 2: 
            key, val = key_val  
            if val in ["True", "False"]:
                kwargs[key] = bool(val)
            else:
                try:
                    kwargs[key] = int(val)
                except ValueError:
                    try: 
                        kwargs[key] = float(val)
                    except ValueError:
                        kwargs[key] = val

        else:
            reg_args.append(arg)
    return reg_args, kwargs

class Completer:
    def __init__(self, words):
        self.words = words
        self.prefix = None
    def complete(self, prefix, index):
        if prefix != self.prefix:
            # we have a new prefix!
            # find all words that start with this prefix
            self.matching_words = [
                w for w in self.words if w.startswith(prefix)
                ]
            self.prefix = prefix
        try:
            return self.matching_words[index]
        except IndexError:
            return None


def cmd_readstream(args, t, active_events):
    
    import textwrap
    from goose import Goose, Configuration
    config = Configuration()
    config.enable_image_fetching = False
    g = Goose(config)
    raw_stream = True

    for arg in args:
        if arg == "articles":
            raw_stream = False

    for event in active_events:
        print event
        if event.query_id.startswith("TS13"):        
            corpus = cuttsum.corpora.EnglishAndUnknown2013()
        elif event.query_id.startswith("TS14"):
            corpus = cuttsum.corpora.SerifOnly2014()
        else:
            raise Exception("Bad query id: {}".format(event.query_id)) 

        if raw_stream is True:
            from cuttsum.trecdata import SCChunkResource
            si_iter = SCChunkResource().streamitem_iter(event, corpus)
        else:
            from cuttsum.pipeline import ArticlesResource
            si_iter = ArticlesResource().streamitem_iter(event, corpus)

        for hour, path, si in si_iter:
            if si.body.clean_visible is not None:
                print si.stream_id
                try:
                    text_height = t.height-4
                    #n_chars = t.
                    article = g.extract(raw_html=si.body.clean_html)
                    lines = textwrap.wrap(article.cleaned_text)
                    idx = 0
                    while 1:
                        print t.clear
                        print "hour:", hour
                        print "title:", article.title
                        print "article:"
                        print "\n".join(lines[idx:idx+text_height])
                    #print article.cleaned_text                
                    
                        with t.cbreak():
                            char = t.inkey()
                            if char == "i" and idx > 0:
                                idx -= 1 #idx - 1 if idx > 0 else 0
                            elif char == "k" and idx + text_height < len(lines):
                                idx += 1 
                            elif char == "l":
                                break

                except Exception, e:
                    print e
                    continue                



def cmd_oracle(args, t, active_events):
    import cuttsum.judgements
    print t.clear

    all_matches = cuttsum.judgements.get_merged_dataframe()
    for event in active_events:
        print event
        matches = all_matches[all_matches["query id"]  == event.query_id]
        matching_doc_ids = set(matches["document id"].tolist())
        
        print matches[0:10]
        if event.query_id.startswith("TS13"):        
            corpus = cuttsum.corpora.EnglishAndUnknown2013()
        elif event.query_id.startswith("TS14"):
            corpus = cuttsum.corpora.SerifOnly2014()
        else:
            raise Exception("Bad query id: {}".format(event.query_id)) 
        
        
        
        from cuttsum.trecdata import SCChunkResource

        chunks_res = SCChunkResource()
        for hour, path, si in chunks_res.streamitem_iter(event, corpus):
            print si.stream_id, hour
            if si.stream_id in matching_doc_ids:
                print si.stream_id, si.stream_time, hour
                for match in matches[matches["document id"] == si.stream_id].iterrows():
                    print match[["nugget text", "update text"]]

            #if si.body.clean_visible is not None:
        
                with t.cbreak():
                    t.inkey()

def main():
    words = ["exit", "select", "deselect", "status", "timeline", "readstream", 
        "oracle", "jobmanager",
        "cuttsum.trecdata.UrlListResource", "cuttsum.trecdata.SCChunkResource"]
    completer = Completer(words)
    readline.parse_and_bind("tab: complete")
    readline.set_completer(completer.complete)
    active_events = []
    all_events = cuttsum.events.get_events() 
    for e in all_events:
        print e.query_num
    t = Terminal()

    with t.fullscreen():
        while 1:
            home_screen(t, active_events)
            with t.location(0, t.height-1):
                cmd = raw_input(">>> ")
            args = cmd.split(" ")

            if args[0] == "select" and len(args) >= 2:
                cmd_select_events(
                    args[1:], active_events, all_events)
            elif args[0] == "deselect" and len(args) >= 2:
                active_events = cmd_deselect_events(
                    args[1:], active_events)
            elif args[0] == "timeline":
                print_event_timelines(t, active_events)
            elif args[0] == "status":
                cmd_status(args[1:], t, active_events)
            elif args[0] == "run":
                cmd_run(args[1:], t, active_events)

            elif args[0] == "readstream":
                cmd_readstream(args[1:], t, active_events)
            elif args[0] == "oracle":
                cmd_oracle(args[1:], t, active_events)
            elif args[0] == "jobmanager":
                from app.job_manager import start_job_manager
                start_job_manager(args[1:], t, active_events)
            elif args[0] == "exit":
                break
            #with t.location(0,0):



#t = Terminal()
#home_screen(t, active_events)
if __name__ == u"__main__":
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD   # get MPI communicator object
        rank = comm.rank
        if rank == 0:
            main()

        elif rank == 1:
            from app.job_manager import manager_process
            manager_process()

        else:
            from app.job_manager import worker_process
            worker_process()



    except Exception, msg:
        print msg

#main()
    
#print_event_timelines()
