import os
import sys
import argparse
from cuttsum.event import read_events_xml
from cuttsum.nuggets import read_nuggets_tsv
from cuttsum.util import gen_dates
import cuttsum.wtmf
import streamcorpus as sc
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np

def main():

    event_file, rc_dir, event_title, nuggets_tsv, ss_params, ofile = parse_args()
    ss_model, ss_vocab, ss_dims = ss_params
    event = load_event(event_title, event_file)
    nuggets = read_nuggets_tsv(nuggets_tsv, filter_query_id=event.query_id)
    hours = [dth for dth in gen_dates(event.start, event.end)]
    print u"Found", len(nuggets), u"nuggets."

    print u"Loading sentence-sim model..."
    wmat_model = cuttsum.wtmf.load_model(ss_model, ss_vocab, latent_dims=ss_dims)    
    nugget_lvecs = wmat_model.factor_unicode([n.text for n in nuggets])    

    
    meta_data = []
    unicodes = []
    print u"Loading sentence data..."
    nhours = len(hours)
    for h, hour in enumerate(hours, 1):
        chunk = os.path.join(rc_dir, u'{}.sc.gz'.format(hour))

        for si_idx, si in enumerate(sc.Chunk(path=chunk)):
            if u'article-clf' not in si.body.sentences:
                continue
            sent_idx_map = {}
            for idx, sent in enumerate(si.body.sentences[u'serif']):
                sent_idx_map[sentence_uni(sent)] = idx
            for sent in si.body.sentences[u'article-clf']:
                uni = sentence_uni(sent)
                meta_data.append((hour, si.stream_id, sent_idx_map[uni]))
                unicodes.append(uni)

    print u"Computing similarities..."
    sent_lvecs = wmat_model.factor_unicode(unicodes)    
    S = cosine_similarity(sent_lvecs, nugget_lvecs)
    S = np.ma.masked_array(S, np.isnan(S))

    Szmuv = (S - S.mean(axis=0)) / S.std(axis=0)
    M = np.amax(Szmuv, axis=1)
    m = np.amin(Szmuv, axis=1)
    U = np.mean(Szmuv, axis=1)
    T = np.sum(Szmuv, axis=1)

    ### WRITE TSV HEADER AND DATA ###
    print u"Writing to", ofile
    header = 'date-hour\tstream-id\tsent-id\tmax-sim\tmin-sim' + \
             '\tmean-sim\ttotal-sim'
    for i in range(ss_dims):
        header += '\tlv{}'.format(i)

    with open(ofile, 'w') as f:
        f.write(header)
        f.write('\n') 
        for idx, meta_datum in enumerate(meta_data):
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(meta_datum[0], meta_datum[1],
                                                    meta_datum[2], M[idx], m[idx], U[idx]))
            for c in range(ss_dims):
                f.write('\t{}'.format(sent_lvecs[idx,c]))
            f.write('\n')
            f.flush()
 

def sentence_uni(sent):
    return u' '.join(token.token.decode(u'utf-8') for token in sent.tokens)


def get_active_nuggets(hour, nuggets, lvecs):
    act_nugs = []
    idx = 0
    for nugget in nuggets:
        if nugget.timestamp.strftime("%Y-%m-%d-%H") <= hour:
            idx += 1
        else:
            break
    if idx > 0:
        return lvecs[0:idx,:]
    else:
        return None

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

    parser.add_argument('-s', '--sent-sim-model',
                        help=u'Location of sentence sim model',
                        type=unicode, required=True)

    parser.add_argument('-v', '--sent-sim-vocab',
                        help=u'Location of sentence sim vocab',
                        type=unicode, required=True)

    parser.add_argument('-d', '--sent-sim-dims',
                        help=u'Sentence-sim model dimensions',
                        type=int, required=True)

    parser.add_argument('-o', '--output-file',
                        help=u'Location to write sims',
                        type=unicode, required=True)

    args = parser.parse_args()
    event_file = args.event_file
    rc_dir = args.rel_chunks_dir
    event_title = args.event_title
    nuggets_tsv = args.nuggets_tsv
    ss_model = args.sent_sim_model
    ss_vocab = args.sent_sim_vocab
    dims = args.sent_sim_dims
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

    if not os.path.exists(nuggets_tsv) or os.path.isdir(nuggets_tsv):
        sys.stderr.write((u'--nuggets-tsv argument {} either does not' \
                          + u' exist or is a directory!\n').format(
                            nuggets_tsv))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(ss_model) or os.path.isdir(ss_model):
        sys.stderr.write((u'--sent-sim-model argument {} either does not' \
                          + u' exist or is a directory!\n').format(
                            ss_model))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(ss_vocab) or os.path.isdir(ss_vocab):
        sys.stderr.write((u'--sent-sim-vocab argument {} either does not' \
                          + u' exist or is a directory!\n').format(
                            ss_vocab))
        sys.stderr.flush()
        sys.exit()

    return (event_file, rc_dir, event_title, nuggets_tsv, 
            (ss_model, ss_vocab, dims), ofile)

if __name__ == '__main__':
    main()
