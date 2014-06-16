import os
import argparse
import sys
import cPickle as pickle
from cuttsum.event import read_events_xml
from cuttsum.judgements import read_matches_tsv
import re
import streamcorpus as sc
from collections import defaultdict

def main():

    event_xml, chunks_dir, event_title, matches_file = parse_args()

    print '2013 Relevance Evaluation'
    print '========================='

    sys.stdout.write('Reading event "{}" '.format(event_title))
    sys.stdout.write('from event xml: {}...\t'.format(event_xml))
    event = load_event(event_title, event_xml)
    print 'OK\n'

    sys.stdout.write('Reading query id "{}" '.format(event.query_id))
    sys.stdout.write('from matches tsv: {}...\t'.format(matches_file))
    
    matches = defaultdict(list)
    for match in read_matches_tsv(matches_file, event.query_id):
        items = match.update_id.split('-')
        stream_id = items[0] + '-' + items[1]
        sent_id = int(items[2])
        matches[stream_id].append((sent_id, match))
    for match in matches.iterkeys():
        matches[match].sort(key=lambda x: x[0])

    print 'OK\n'

    num_gold = len(matches)
    num_matches = 0
    num_si = 0   
    msg = sc.StreamItem_v0_2_0
    for path in find_chunks(chunks_dir, event):
        #print path
        for si in sc.Chunk(path=path, message=msg):
            #print si.stream_id
            print si.abs_url
            num_si += 1
            if si.stream_id in matches:
                print si.stream_id, si.abs_url
                for sidx, match in matches[si.stream_id]:
                    print_sentence(si.body.sentences['lingpipe'][sidx])
                num_matches += 1

    print 'Precision:', num_matches / float(num_si)
    print 'Recall:', num_matches / float(num_gold)



def print_sentence(sentence):
    for token in sentence.tokens:
        print token.token,
    print

def find_chunks(chunks_dir, event):
    unordered_chunks = []
    event_name = event.title.replace(' ', '_')
    for fname in os.listdir(chunks_dir):
        if event_name in fname:
            path = os.path.join(chunks_dir, fname)
            m = re.search(event_name + r'.*?\.(\d+)\.(gz|xz)', path, re.I)
            if m:
                ts = int(m.group(1))
                unordered_chunks.append((ts, path))
    chunks = [i[1] for i in sorted(unordered_chunks, key=lambda x: x[0])]             
    return chunks

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

    parser.add_argument('-c', '--chunks-dir',
                        help=u'Relevant chunks directory',
                        type=str, required=True)

    parser.add_argument('-t', '--event-title',
                        help=u'Event title',
                        type=unicode, required=True)

    parser.add_argument('-m', '--matches-file',
                        help=u'2013 Matches File.',
                        type=unicode, required=True)
   
    args = parser.parse_args()
    event_file = args.event_file
    chunks_dir = args.chunks_dir
    event_title = args.event_title
    matches_file = args.matches_file

    if not os.path.exists(event_file) or os.path.isdir(event_file):
        sys.stderr.write((u'--event-file argument {} either does not exist' \
                          + u' or is a directory!\n').format(event_file))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(chunks_dir) or not os.path.isdir(chunks_dir):
        sys.stderr.write((u'--chunks-dir argument {} either does not exist' \
                          + u' or is not a directory!\n').format(chunks_dir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(matches_file) or os.path.isdir(matches_file):
        sys.stderr.write((u'--matches-file argument {} either does not exist' \
                          + u' is not a directory!\n').format(matches_file))
        sys.stderr.flush()
        sys.exit()

    return event_file, chunks_dir, event_title, matches_file

if __name__ == '__main__':
    main()
