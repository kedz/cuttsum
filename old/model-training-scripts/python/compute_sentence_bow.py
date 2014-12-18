import os
import sys
import argparse
from cuttsum.event import read_events_xml
from cuttsum.nuggets import read_nuggets_tsv
from cuttsum.util import gen_dates
import streamcorpus as sc
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import codecs

def main():
    event_file, rc_dir, event_title, ofile = parse_args()
    event = load_event(event_title, event_file)
    hours = [dth for dth in gen_dates(event.start, event.end)]
    num_hours = len(hours)

    meta_data = []
    bow_dicts = []

    for h, hour in enumerate(hours, 1):
        path = os.path.join(rc_dir, '{}.sc.gz'.format(hour))
        for si in sc.Chunk(path=path):
            uni2id = {}
            for sid, sentence in enumerate(si.body.sentences[u'serif'], 0):
                uni2id[sentence_uni(sentence)] = sid
            
            for sent in si.body.sentences[u'article-clf']:
                bow_dict = {}
                for token in sent.tokens:
                    t = token.token.decode(u'utf-8').lower()
                    bow_dict[t] = 1
                bow_dicts.append(bow_dict)
                uni = sentence_uni(sent)
                sent_id = uni2id[uni]
                meta_data.append((hour, si.stream_id, sent_id, uni))

    vctr = DictVectorizer()
    X = vctr.fit_transform(bow_dicts)

    with codecs.open(ofile, 'w', 'utf-8') as f:
        for i, (hour, stream_id, sent_id, uni) in enumerate(meta_data):
            uni = uni.replace(u'\n', u' ').replace(u'\t', u' ')
            f.write(u'{}\t{}\t{}\t{}\t'.format(hour, stream_id, sent_id, uni))
            x = u' '.join([unicode(col) for col in X[i,:].indices])
            f.write(x)
            f.write(u'\n')
            f.flush()

def sentence_uni(sent):
    return u' '.join(token.token.decode(u'utf-8') for token in sent.tokens)
 
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

    parser.add_argument('-t', '--event-title',
                        help=u'Event title',
                        type=unicode, required=True)

    parser.add_argument('-o', '--output-file',
                        help=u'Location to write sims',
                        type=unicode, required=True)

    args = parser.parse_args()
    event_file = args.event_file
    rc_dir = args.rel_chunks_dir
    event_title = args.event_title
    ofile = args.output_file
    
    odir = os.path.dirname(ofile)
    if odir != u'' and not os.path.exists(odir):
        os.makedirs(odir)

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

    return (event_file, rc_dir, event_title, ofile)

if __name__ == '__main__':
    main()
