import os
import argparse
from cuttsum.event import read_events_xml
from cuttsum.util import gen_dates, memory, resident, stacksize
import streamcorpus as sc
from multiprocessing import Pool, Value
from datetime import datetime, timedelta
import re
from collections import defaultdict
import sys
import cPickle as pickle
import pympler
import subprocess
from time import sleep
import time
import gc

stop_flag = Value('i', 0)

def main():
    args = parse_args()
    event_xml, ws, chunkurls_file, event_title = args[0:4]
    output_chunk, output_stats, n_threads = args[4:]

    print 'TREC TS Static Relevance Extractor'
    print '=================================='
    
    sys.stdout.write('Reading event "{}" '.format(event_title))
    sys.stdout.write('from event xml: {}...\t'.format(event_xml))
    sys.stdout.flush()
    event = load_event(event_title, event_xml)
    print 'OK\n'

    start_dt = event.start - timedelta(hours=24)
    end_dt = event.end
    #end_dt = start_dt + timedelta(hours=20)
    query = tuple(event.query + [w.lower() for w in event.title.split(' ')])

    
    sys.stdout.write('Reading urls from {}...\t'.format(chunkurls_file))
    sys.stdout.flush()
    global stop_flag
    urls = read_urls(chunkurls_file, start_dt, end_dt)
    jobs = [(query, url, ws) for url in urls]
    print 'OK'

    print 'SYSTEM INFO'
    print '==========='
    print 'Workspace Location:', ws
    print 'Output Chunk:', output_chunk
    print 'Output Stats Dir:', output_stats
    print 'n-threads:', n_threads 
    print

    print 'EVENT INFO'
    print '=========='
    print 'Event Title:', event.title
    print 'Event Type:', event.type
    print 'Date Range: {} -- {}'.format(event.start, event.end)
    print 'Query Words:', ', '.join(query)
    print

    tot_num_si_rel = defaultdict(int)
    tot_num_si_irrel = defaultdict(int)
    tot_df_rel = {}
    tot_df_irrel = {}
    tot_wc_rel = {}
    tot_wc_irrel = {}
    num_rel = 0
    num_irrel = 0

    relevant_items = []
    #jobs = make_jobs(chunks_dir, start_dt, end_dt, query)
    njobs = len(jobs)

    
    latest_dt = start_dt
    
    from math import floor
    from pympler.asizeof import asizeof

    if n_threads == 1:
        results = single_threaded_worker(jobs)
    else:

        pool = Pool(n_threads, maxtasksperchild=2)
        results = pool.imap(worker, jobs)

#    alerts = range(5,105,5)
#    first_alert = alerts[0]   
#    last_alert = alerts[-1]
    first_alert = True




    for i, result in enumerate(results, 1):
        per = 100.0 * float(i) / njobs

        if sys.stdout.isatty():
            sys.stdout.write('{:2.2f}% complete\r'.format(per))
            sys.stdout.flush()
        print i, memory()/float(1024**3), resident()/float(1024**3), 
        print stacksize()/float(1024**3)
        print 

        num_si_rel, num_si_irrel, df_rel, df_irrel = result[0:4]
        wc_rel, wc_irrel, lnum_rel, lnum_irrel, relevant_sis = result[4:]
       
        if i == 1:
            append = False
        else:
            append = True



        pickle_stats(tot_num_si_rel, 'num_si_rel.p', output_stats, append)
        pickle_stats(tot_num_si_irrel, 'num_si_irrel.p', 
                     output_stats, append)
        pickle_stats(tot_df_rel, 'df_rel.p', output_stats, append)
        pickle_stats(tot_df_irrel, 'df_irrel.p', output_stats, append)
        pickle_stats(tot_wc_rel, 'wc_rel.p', output_stats, append)
        pickle_stats(tot_wc_irrel, 'wc_irrel.p', output_stats, append)




        num_rel += lnum_rel
        num_irrel += lnum_irrel

        #update_stats(num_si_rel, num_si_irrel,
        #             df_rel, df_irrel, wc_rel, wc_irrel, 
        #             tot_num_si_rel, tot_num_si_irrel,
        #             tot_df_rel, tot_df_irrel, tot_wc_rel, tot_wc_irrel)
 

        for dt, si in relevant_sis:
            if dt > latest_dt:
                latest_dt = dt
        
        relevant_items.extend(relevant_sis)


        total = 0                                    
#        tot_num_si_rel_size = asizeof(tot_num_si_rel) / float(1024**2) 
#        total += tot_num_si_rel_size
#        tot_num_si_irrel_size = asizeof(tot_num_si_irrel) / float(1024**2)
#        total += tot_num_si_irrel_size
#        tot_df_rel_size = asizeof(tot_df_rel) / float(1024**2)
#        total += tot_df_rel_size
#        tot_df_irrel_size = asizeof(tot_df_irrel) / float(1024**2)
#        total += tot_df_irrel_size
#        tot_wc_rel_size = asizeof(tot_wc_rel) / float(1024**2)
#        total += tot_wc_rel_size
 #       tot_wc_irrel_size = asizeof(tot_wc_irrel) / float(1024**2)
#        total += tot_wc_irrel_size
        relevant_items_size = asizeof(relevant_items) / float(1024**2)
        total += relevant_items_size

        print 'Memory'
        print '======'
#        print '{:25} : {:7.2f}mb'.format('tot_num_si_rel', 
#                                         tot_num_si_rel_size)
#        print '{:25} : {:7.2f}mb'.format('tot_num_si_irrel', 
#                                         tot_num_si_irrel_size)
#        print '{:25} : {:7.2f}mb'.format('tot_df_rel', tot_df_rel_size)
#        print '{:25} : {:7.2f}mb'.format('tot_df_irrel', tot_df_irrel_size)
#        print '{:25} : {:7.2f}mb'.format('tot_wc_rel', tot_wc_rel_size)
#        print '{:25} : {:7.2f}mb'.format('tot_wc_irrel', tot_wc_irrel_size)
        print '{:25} : {:7.2f}mb'.format('relevant_items',
                                         relevant_items_size)
        print '-' * 25
        print '{:25} : {:7.2f}mb'.format('Total', total)
        print 


                
        if total / float(1024) > .1 or i == njobs:
            stop_flag.value = 1

            print "CLEARING SOME STUFF"
            if first_alert is True:
                append = False
                first_alert = False
            else:
                append = True

            if i == njobs:
                clear_cache = True
            else:
                clear_cache = False




            print '\r',
            print datetime.now(), '|| Relevant:', num_rel, 
            print ' || Irrelevant:', num_irrel
            print
            print 'Latest Stream Item Datetime'
            print '==========================='
            print latest_dt
            print
       
            
#            tot_num_si_rel = None
#            tot_num_si_irrel = None
#            tot_df_rel = None
#            tot_df_irrel = None
#            tot_wc_rel = None
#            tot_wc_irrel = None
#            gc.collect()
#            tot_num_si_rel = defaultdict(int)
#            tot_num_si_irrel = defaultdict(int)
#            tot_df_rel = {}
#            tot_df_irrel = {}
#            tot_wc_rel = {}
#            tot_wc_irrel = {}
#            print
#
            relevant_items = clear_rel_si_cache(relevant_items, output_chunk,
                                                latest_dt, clear_cache)

            stop_flag.value = 0
    
        print 'comp'
    pickle_stats(event, 'event.p', output_stats, False)

    print '\r100.00% complete' 


def clear_rel_si_cache(relevant_items, output_chunk, latest_dt, clear_cache):
    keep_items = []
    write_items = []
     
    sys.stdout.write('Sorting relevant Stream Items...\t')
    sys.stdout.flush()
    relevant_items.sort(key=lambda x: x[0])
    print 'OK'
    
    cutoff = latest_dt - timedelta(hours=1)
    for dt, si in relevant_items:
        if dt < cutoff or clear_cache is True:
            write_items.append(si)
        else:
            keep_items.append((dt, si))
    if len(write_items) > 0:
        p, ext = os.path.splitext(output_chunk)
        utc = int((latest_dt - datetime(1970,1,1)).total_seconds())
        path = p + '.{}'.format(utc) + ext
        if os.path.exists(path):
            os.remove(path)
        print 'Writing relevant items to file: {}...\n'.format(path)
        
        msg = sc.StreamItem_v0_2_0
        ochunk = sc.Chunk(path=str(path), message=msg, mode='wb')
        for si in write_items:
            print si.stream_time.zulu_timestamp, si.stream_id
            ochunk.add(si)
        ochunk.close()

    print 
    return keep_items

#    for dt, si in relevant_items:
#        ochunk.add(si)
#    ochunk.close()

    
#    import matplotlib
#    matplotlib.use('QT4Agg')
#    import matplotlib.pyplot as plt
#    import matplotlib.dates as mdates
#    from itertools import izip, cycle 
#
#    
#    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#    rel_df = {}
#    conver = mdates.strpdate2num('%Y-%m-%d-%H')
#    hours = [conver(dt) for dt in gen_dates(start_dt, end_dt)]
#    for qw, col in izip(query, colors):
#        df_rel[qw] = []
#        for dt in gen_dates(start_dt, end_dt):
#            if dt not in tot_df_rel:
#                df_rel[qw].append(0)
#            else:
#                df_rel[qw].append(tot_df_rel[dt][qw])
#        plt.plot_date(x=hours, y = df_rel[qw], fmt=col+'-', label=qw)
#
#    plt.legend()
#    plt.grid(True)
#    plt.show()

def pickle_stats(obj, name, odir, append):
    if not os.path.exists(odir):
        os.makedirs(odir)
    pfname = os.path.join(odir, name)
    sys.stdout.write('Pickling stats: {}...\t'.format(pfname))
    sys.stdout.flush()
    if append:
        with open(pfname, 'ab') as p:
            pickle.dump(obj, p)
    else:    
        with open(pfname, 'wb') as p:
            pickle.dump(obj, p)
    print 'OK'

def update_stats(num_si_rel, num_si_irrel,
                 df_rel, df_irrel, wc_rel, wc_irrel, 
                 tot_num_si_rel, tot_num_si_irrel,
                 tot_df_rel, tot_df_irrel, tot_wc_rel, tot_wc_irrel):
 
    for key, val in num_si_rel.iteritems():
        tot_num_si_rel[key] += val           
    for key, val in num_si_irrel.iteritems():
        tot_num_si_irrel[key] += val           

    for key, wcs in df_rel.iteritems():
        if key not in tot_df_rel:
            tot_df_rel[key] = defaultdict(int)
        for word, count in wcs.iteritems():
            tot_df_rel[key][word] += count

    for key, wcs in df_irrel.iteritems():
        if key not in tot_df_irrel:
            tot_df_irrel[key] = defaultdict(int)
        for word, count in wcs.iteritems():
            tot_df_irrel[key][word] += count

    for key, wcs in wc_rel.iteritems():
        if key not in tot_wc_rel:
            tot_wc_rel[key] = defaultdict(int)
        for word, count in wcs.iteritems():
            tot_wc_rel[key][word] += count
   
    for key, wcs in wc_irrel.iteritems():
        if key not in tot_wc_irrel:
            tot_wc_irrel[key] = defaultdict(int)
        for word, count in wcs.iteritems():
            tot_wc_irrel[key][word] += count


def single_threaded_worker(jobs):
    for job in jobs:
        yield worker(job)

def worker(args):
    query, url, ws  = args
    path = download_url(url, ws)

    global stop_flag
    while stop_flag.value != 0:
        sleep(1)

    msg = sc.StreamItem_v0_2_0
    
    relevant_sis = []

    num_si_rel = defaultdict(int)
    num_si_irrel = defaultdict(int)
    df_rel = {}
    df_irrel = {}
    wc_rel = {}
    wc_irrel = {}    
    num_rel = 0
    num_irrel = 0 

    if path is not None:

        for si in sc.Chunk(path=path, message=msg):

            st = sc.make_stream_time(si.stream_time.zulu_timestamp) 
            dt = datetime.utcfromtimestamp(int(st.epoch_ticks))
            dtstr = dt.strftime('%Y-%m-%d-%H')  
      
            if si.body.clean_visible is None:
                relevant = False
            else:
                relevant = check_relevance(query, si.body.clean_visible)

            if relevant is True:
                num_rel += 1
                if dtstr not in df_rel:
                    df_rel[dtstr] = defaultdict(int)
                if dtstr not in wc_rel:
                    wc_rel[dtstr] = defaultdict(int)

                relevant_sis.append((dt, si))
                num_si_rel[dtstr] += 1
                
                for sentence in si.body.sentences['lingpipe']:
                    for token in sentence.tokens:
                        df_rel[dtstr][token.token.lower()] = 1    
                        wc_rel[dtstr][token.token.lower()] += 1
            else:
                num_irrel += 1
                if dtstr not in df_irrel:
                    df_irrel[dtstr] = defaultdict(int)
                if dtstr not in wc_irrel:
                    wc_irrel[dtstr] = defaultdict(int)

                num_si_irrel[dtstr] += 1

                for sentence in si.body.sentences['lingpipe']:
                    for token in sentence.tokens:
                        df_irrel[dtstr][token.token.lower()] = 1    
                        wc_irrel[dtstr][token.token.lower()] += 1

        os.remove(path)

    return (num_si_rel, num_si_irrel, df_rel, 
            df_irrel, wc_rel, wc_irrel, num_rel, num_irrel, relevant_sis)

def download_url(url, ws):
    url_toks = url.split('/')
    fpath = os.path.join(ws, url_toks[-1])
    xzpath = os.path.splitext(fpath)[0]
    if os.path.exists(xzpath):
        return xzpath

    tries = 0
    success = False
    while tries < 3 and success is False:

        p1 = subprocess.Popen(['curl', '-o' , fpath, url])

        elapsed = 0
        last_time = time.time()
        print p1.poll()
        while p1.poll() is None and elapsed < 20:
            time.sleep(1)
            elapsed = time.time() - last_time 
        
        if p1.poll() is None or not os.path.exists(fpath):
            p1.terminate()
            tries += 1
            print 'Tries', tries
            time.sleep(2)
        else:
            success = True

    if success is False:

        print
        print "KILLING:", url
        return None




    with open(xzpath, 'wb') as f, open(os.devnull, "w") as fnull:
        #subprocess.call(['wget', '-P', ws, '-t', '0', url],
        #                stdout=fnull, stderr=fnull)
        
        subprocess.call(['gpg', '--decrypt', fpath],
                              stdout=f, stderr=fnull)




#    thread = threading.Thread(target=runner)
#    thread.start()
#    thread.join(10)
#    if thread.is_alive():
#        print 'Failed to download:', url
            

    os.remove(fpath)
    return xzpath

def check_relevance(query_words, clean_html):
    for word in query_words:
        if not re.search(word, clean_html, re.I):
            return False
    return True

def make_jobs(chunks_dir, start_dt, end_dt, query):
    jobs = []
    for yyyymmddhh in gen_dates(start_dt, end_dt):
        ddir = os.path.join(chunks_dir, yyyymmddhh)
        if os.path.exists(ddir):
            for fname in os.listdir(ddir):
                jobs.append((query, os.path.join(ddir, fname)))
    return jobs

def load_event(event_title, event_xml):
    events = read_events_xml(event_xml)
    for event in events:
        if event_title == event.title:
            return event
    raise ValueError(("No event title matches \"{}\" " \
                      + "in file: {}").format(event_title, event_xml))

def read_urls(urls_file, start_date, end_date):
    good_urls = []
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f.readlines()]
        for url in urls:
            url_toks = url.split('/')    

            date_str = '-'.join(url_toks[-2].split('-')[:-1]) + \
                'T00:00:00.000000Z'
            etstr = sc.make_stream_time(date_str).epoch_ticks
            urldate = datetime.utcfromtimestamp(etstr)
            
            if start_date <= urldate and urldate < end_date:
                good_urls.append(url)
    
    return good_urls

                                                                   
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--event-file',
                        help=u'Event xml file.',
                        type=unicode, required=True)

    parser.add_argument('-w', '--workspace-dir',
                        help=u'Workspace dir',
                        type=str, required=True)

    parser.add_argument('-c', '--chunk-urls',
                        help=u'Files with chunk urls.',
                        type=str, required=True)

    parser.add_argument('-t', '--event-title',
                        help=u'Event title',
                        type=unicode, required=True)

    parser.add_argument('-o', '--output-chunk',
                        help=u'Chunk file to write relevant StreamItems to.',
                        type=str, required=True)

    parser.add_argument('-s', '--output-stats',
                        help=u'Location to write stats/plots.',
                        type=unicode, required=True)

    parser.add_argument('-nt', '--n-threads',
                        help=u'Number of threads.',
                        type=int, required=False, default=1)
    
    args = parser.parse_args()
    event_file = args.event_file
    ws = args.workspace_dir
    chunkurls_file = args.chunk_urls
    event_title = args.event_title
    output_chunk_file = args.output_chunk
    output_stats_dir = args.output_stats
    n_threads = args.n_threads    

    if not os.path.exists(event_file) or os.path.isdir(event_file):
        import sys
        sys.stderr.write((u'--event-file argument {} either does not exist' \
                          + u' or is a directory!\n').format(event_file))
        sys.stderr.flush()
        sys.exit()

    if ws != '' and not os.path.exists(ws):
        os.makedirs(ws)

    odir = os.path.dirname(output_chunk_file)
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)

    if output_stats_dir != '' and not os.path.exists(output_stats_dir):
        os.makedirs(output_stats_dir)

    if not os.path.exists(chunkurls_file) or os.path.isdir(chunkurls_file):

        sys.stderr.write((u'--chunk-urls argument {} either does not exist' \
                          + u' or is a directory!\n').format(chunk_urls))
        sys.stderr.flush()
        sys.exit()

    return (event_file, ws, chunkurls_file, event_title, 
            output_chunk_file, output_stats_dir, n_threads)

if __name__ == '__main__':
    main()
