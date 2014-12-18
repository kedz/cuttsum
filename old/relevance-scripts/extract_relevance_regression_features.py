import argparse
import os
import sys
from cuttsum.event import read_events_xml
from cuttsum.nuggets import read_nuggets_tsv
from cuttsum.util import gen_dates
from datetime import datetime, timedelta
import gzip
import streamcorpus as sc
from math import log

def main():
    
    args = parse_args()
    event_file, rc_dir, event_title, nuggets_tsv = args[0:4]
    doc_freqs, word_freqs = args[4:]

    print 'Generating Regression Features'
    print '=============================='
    event = load_event(event_title, event_file)
    nuggets = read_nuggets_tsv(nuggets_tsv, filter_query_id=event.query_id)
    hours = [dth for dth in gen_dates(event.start, event.end)]

    worker((rc_dir, nuggets, hours, event, doc_freqs, word_freqs))


def worker(args):
    rc_dir, nuggets, hours, event, doc_freqs, word_freqs = args
    msg = sc.StreamItem_v0_2_0

    for hour in hours:
        active_nuggets = get_active_nuggets(hour, nuggets)
        if len(active_nuggets) == 0:
            continue
        hour_m5 = get_previous_hour(hour, 5)
        hour_m10 = get_previous_hour(hour, 10)
        num_docs = doc_count(doc_freqs, hour)
        num_m5_docs = doc_count(doc_freqs, hour_m5)
        num_m10_docs = doc_count(doc_freqs, hour_m10)
        hour_wc = read_tfdf(word_freqs, hour)
        hour_m5_wc = read_tfdf(word_freqs, hour_m5)
        hour_m10_wc = read_tfdf(word_freqs, hour_m10)
        chunk = os.path.join(rc_dir, '{}.sc.gz'.format(hour))
        
        for si in sc.Chunk(path=chunk, message=msg):
            doc_wc = make_doc_wordcounts(si)
            
            for sentence in si.body.sentences['lingpipe']:
                avg_tfidf = compute_avg_tfidf(sentence, doc_wc, 
                                              hour_wc, num_docs)
                avg_m5_tfidf = compute_avg_tfidf(sentence, None, 
                                                 hour_m5_wc, num_m5_docs)
                avg_m10_tfidf = compute_avg_tfidf(sentence, None, 
                                                  hour_m10_wc, num_m10_docs)
                delta_m5_tfidf = avg_tfidf - avg_m5_tfidf
                delta_m10_tfidf = avg_tfidf - avg_m10_tfidf
                
                tokens = [token.token for token in sentence.tokens]
                print avg_tfidf, avg_m5_tfidf, avg_m10_tfidf, ' '.join(tokens)

        print hour, hour_m5, num_m5_docs, hour_m10
        sys.exit()

##        print "============="
#        for nugget in active_nuggets:
#            print nugget.timestamp, nugget.text

def compute_avg_tfidf(sentence, doc_wc, hour_wc, num_docs):
    if num_docs == 0:
        return 0.0
    ntokens = len(sentence.tokens)
    if ntokens == 0:
        return 0.0
    total_tfidf = 0
    for token in sentence.tokens:
        t = token.token.decode('utf-8')
        if t not in hour_wc:
            continue
        if doc_wc is not None:
            tf = doc_wc.get(t, 0)
        else:
            tf = hour_wc[t].get('wf', 0) / float(hour_wc[t].get('df', 1))

        idf = log(num_docs / float(hour_wc[t].get('df', 1)))
        tfidf = (1 + log(tf)) * idf
        total_tfidf += tfidf
    return total_tfidf / float(ntokens)



def make_doc_wordcounts(si):
    counts = {}
    for sentence in si.body.sentences['lingpipe']:
        for token in sentence.tokens:
            t = token.token.decode('utf-8')
            counts[t] = counts.get(t, 0) + 1
    return counts

def read_tfdf(wfdir, hour):
    counts = {}
    path = os.path.join(wfdir, '{}.txt.gz'.format(hour))
    with gzip.open(path, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            word = items[0].decode('utf-8')
            wf = int(items[1])
            df = int(items[2])
            counts[word] = {'wf':wf, 'df':df}
    return counts


def doc_count(dfdir, hour):
    path = os.path.join(dfdir, '{}.txt'.format(hour))
    with open(path, 'r') as f: return int(f.readline().strip())


def get_previous_hour(current_hour, minus):
    year, month, day, hour = current_hour.split('-')
    dt = datetime(int(year), int(month), int(day), int(hour))
    return (dt - timedelta(hours=minus)).strftime('%Y-%m-%d-%H')

      


def get_active_nuggets(hour, nuggets):
    act_nugs = []
    for nugget in nuggets:
        if nugget.timestamp.strftime("%Y-%m-%d-%H") <= hour:
            act_nugs.append(nugget)
    return act_nugs

def load_event(event_title, event_xml):
    events = read_events_xml(event_xml)
    for event in events:
        if event_title == event.title:
            return event
    raise ValueError(("No event title matches \"{}\" " \
                      + "in file: {}").format(event_title, event_xml))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--event-file',
                        help=u'Event xml file.',
                        type=unicode, required=True)

    parser.add_argument('-r', '--rel-chunks-dir',
                        help=u'Relevance Chunks dir',
                        type=str, required=True)

    parser.add_argument('-n', '--nuggets-tsv',
                        help=u'Nuggets tsv file',
                        type=str, required=True)

    parser.add_argument('-t', '--event-title',
                        help=u'Event title',
                        type=unicode, required=True)

    parser.add_argument('-d', '--doc-frequencies',
                        help=u'Directory of date-hour doc frequencies',
                        type=unicode, required=True)

    parser.add_argument('-w', '--word-frequencies',
                        help=u'Directory of date-hour word frequencies',
                        type=unicode, required=True)

    args = parser.parse_args()
    event_file = args.event_file
    rc_dir = args.rel_chunks_dir
    event_title = args.event_title
    nuggets_tsv = args.nuggets_tsv
    doc_freqs = args.doc_frequencies
    word_freqs = args.word_frequencies

    if not os.path.exists(event_file) or os.path.isdir(event_file):
        sys.stderr.write((u'--event-file argument {} either does not exist' \
                          + u' or is a directory!\n').format(event_file))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(rc_dir) or not os.path.isdir(rc_dir):
        sys.stderr.write((u'--rel-chunks-dir argument {} either does not' \
                          + u' exist or is not a directory!\n').format(rc_dir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(nuggets_tsv) or os.path.isdir(nuggets_tsv):
        sys.stderr.write((u'--nuggets-tsv argument {} either does not' \
                          + u' exist or is a directory!\n').format(
                            nuggets_tsv))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(doc_freqs) or not os.path.isdir(doc_freqs):
        sys.stderr.write((u'--doc-frequencies argument {} either does not' \
                          + u' exist or is not a directory!\n').format(
                            doc_freqs))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(word_freqs) or not os.path.isdir(word_freqs):
        sys.stderr.write((u'--word-frequencies argument {} either does not' \
                          + u' exist or is not a directory!\n').format(
                            word_freqs))
        sys.stderr.flush()
        sys.exit()


    return (event_file, rc_dir, event_title, nuggets_tsv, doc_freqs, word_freqs)

if __name__ == '__main__':
    main()
