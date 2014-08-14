import os
import sys
import argparse
from cuttsum.event import read_events_xml
from cuttsum.nuggets import read_nuggets_tsv
from cuttsum.util import gen_dates
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
import srilm_client
import streamcorpus as sc
import gzip
import re
from datetime import datetime, timedelta
from math import log
from collections import defaultdict

def main():

    event_file, rc_dir, event_title, ofile, ports, cnts_dirs = parse_args()
    wc_dir, dc_dir = cnts_dirs
    event = load_event(event_title, event_file)
    hours = [dth for dth in gen_dates(event.start, event.end)]
    print "Connecting lm clients..."
    dm_lm_score = lm_client_init(ports[0])
    bg_lm3_score = lm_client_init(ports[1][0])
    bg_lm4_score = lm_client_init(ports[1][1])
    bg_lm5_score = lm_client_init(ports[1][2])
    print "Query words:", event.query
    query_matcher = query_term_match_init(event.query)

    wn_terms = wn_synset_terms(event.type)
    print "WordNet synset terms:", wn_terms
    synset_matcher = query_term_match_init(wn_terms)

    tfidfers = []
    preroll = [get_previous_hour(hours[0], i) for i in range(1,6)]
    for hour in preroll:
        tfidfers.append(init_tfidfers(wc_dir, dc_dir, hour, lower=True))
    tfidfers.append(None)

    of = open(ofile, 'w')

    header = "hour\tstream-id\tsent-id\t" \
             + "avg-tfidf\tavg-tfidf-m1\tavg-tfidf-m5\t" \
             + "dm-logprob\tdm-avg-logprob\tbg3-logprob\tbg3-avg-logprob\t" \
             + "bg4-logprob\tbg4-avg-logprob\tbg5-logprob\tbg5-avg-logprob\t" \
             + "query-matches\tsynset-matches\tnum-tokens\tarticle-position\t" \
             + "article-position-rel\tcapsrate\n"
    of.write(header)
    of.flush()

    num_hours = len(hours)
    for h, hour in enumerate(hours, 1):
        tfidfers = [init_tfidfers(wc_dir, dc_dir, hour, lower=True)] \
            + tfidfers[0:-1]
        print "({}/{}) {}".format(h, num_hours, hour)

        path = os.path.join(rc_dir, '{}.sc.gz'.format(hour))
        for si in sc.Chunk(path=path):

            ticks = float(si.stream_time.epoch_ticks)          
            si_datetime = datetime.utcfromtimestamp(ticks)
            tdelta = si_datetime - event.start

            uni2id = {}
            doc_word_counts = defaultdict(int)
            for sid, sentence in enumerate(si.body.sentences[u'serif'], 0):
                uni2id[sentence_uni(sentence)] = sid
                for token in sentence.tokens:
                    t = token.token.decode(u'utf-8').lower()
                    doc_word_counts[t] += 1

            nsents = len(si.body.sentences[u'article-clf'])
            for apos, sent in enumerate(si.body.sentences[u'article-clf'], 1):

                tf_dict = {}
                for token in sent.tokens:
                    t = token.token.decode(u'utf-8').lower()
                    tf_dict[t] = doc_word_counts[t]
                tfidfs_now = tfidfers[0](tf_dict)
                tfidfs_m1 = tfidfers[1](tf_dict)
                tfidfs_m5 = tfidfers[5](tf_dict)

                scores = compute_tfidfs(tfidfs_now, tfidfs_m1, tfidfs_m5)
                avg_tfidf, avg_tfidf_m1, avg_tfidf_m5 = scores

                uni = sentence_uni(sent)
                sent_id = uni2id[uni]
                apos_rel = apos / float(nsents)
                num_tokens = len(sent.tokens)
                caps_rate = get_caps_rate(sent)
                dm_lp, dm_alp = dm_lm_score(uni)
                bg3_lp, bg3_alp = bg_lm3_score(uni)
                bg4_lp, bg4_alp = bg_lm4_score(uni)
                bg5_lp, bg5_alp = bg_lm5_score(uni)
                query_matches = query_matcher(uni)
                synset_matches = synset_matcher(uni)
#                print dm_lp, dm_alp, bg3_lp, bg3_alp, bg4_lp, bg4_alp, bg5_lp, bg5_alp

                dstr = ('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}' \
                        +'\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n').format(
                    hour, si.stream_id, sent_id,
                    avg_tfidf, avg_tfidf_m1, avg_tfidf_m5,  
                    dm_lp, dm_alp, bg3_lp, bg3_alp, bg4_lp, bg4_alp, 
                    bg5_lp, bg5_alp, query_matches, synset_matches,
                    num_tokens, apos, apos_rel, caps_rate)
                of.write(dstr) 
                of.flush()
    of.close()

def get_previous_hour(current_hour, minus):
    year, month, day, hour = current_hour.split('-')
    dt = datetime(int(year), int(month), int(day), int(hour))
    return (dt - timedelta(hours=minus)).strftime('%Y-%m-%d-%H')


def compute_tfidfs(tfidfs_now, tfidfs_m1, tfidfs_m5):
    tot_tfidf_now = 0.0
    tot_tfidf_m1 = 0.0
    tot_tfidf_m5 = 0.0

    for word, tfidf in tfidfs_now.iteritems():
        tot_tfidf_now += tfidf
        tot_tfidf_m1 += tfidf - tfidfs_m1[word]
        tot_tfidf_m5 += tfidf - tfidfs_m5[word]

    if len(tfidfs_now) > 0:
        num_words = float(len(tfidfs_now))
        return (tot_tfidf_now / num_words, tot_tfidf_m1 / num_words,
                tot_tfidf_m5 / num_words)
    else:
        return 0.0, 0.0, 0.0

def init_tfidfers(wc_dir, dc_dir, hour, lower=False):
    dpath = os.path.join(dc_dir, '{}.txt'.format(hour))
    if os.path.exists(dpath):
        f = open(dpath) 
        num_docs = int(f.readline().strip())
        f.close()
    else:
        num_docs = 0

    counts = {}
    path = os.path.join(wc_dir, '{}.txt.gz'.format(hour))
    if os.path.exists(path):
        with gzip.open(path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')

                word = items[0].decode('utf-8')
                if lower is True:
                    word = word.lower()

                wf = int(items[1])
                df = int(items[2])
                if word not in counts:
                    counts[word] = {'wf':wf, 'df':df}
                else:
                    counts[word]['wf'] += wf
                    counts[word]['df'] += df
    
    def avg_tfidf_from_tf_dict(tf_dict):
        tot_tfidf = 0.0
        for word, tf in tf_dict.iteritems():
            tot_tfidf += log(1 + tf)*log(num_docs / float(counts[word]['df']))
        if len(tf_dict) > 0:
            return tot_tfidf / float(len(tf_dict))     

    def tfidf_dict(tf_dict):
        tfidf_dict = {}
        for word, tf in tf_dict.iteritems():
            if word not in counts:
                tfidf_dict[word] = 0.0
            else:
                ltf = log(1 + tf)
                idf = log(num_docs / float(counts[word]['df']))
                tfidf_dict[word] = ltf * idf
        return tfidf_dict

    return tfidf_dict

def sentence_uni(sent):
    return u' '.join(token.token.decode(u'utf-8') for token in sent.tokens)

def lm_client_init(port):

    tokenizer = RegexpTokenizer(r'\w+')
    client = srilm_client.Client(port, False)
    
    def lm_score(sentence_uni):
        words = tokenizer.tokenize(sentence_uni)
        sent_string = u' '.join(words)
        nwords = len(words)
        if nwords > 0:
            score = client.sentence_log_prob(sent_string.encode('utf-8'))
            return score, (score / float(nwords))
        else:
            return 0
    return lm_score

def wn_synset_terms(event_type):
    terms = []
    for synset in wn.synsets(event_type):
        terms.extend([lemma.name.decode('utf-8').replace(u'_', u' ')
                       for lemma in synset.lemmas])
        terms.extend([lemma.name.decode('utf-8').replace(u'_', u' ')
                      for synset in synset.hypernyms()
                      for lemma in synset.lemmas])
        terms.extend([lemma.name.decode('utf-8').replace(u'_', u' ')
                      for synset in synset.hyponyms()
                      for lemma in synset.lemmas])
    return terms

def query_term_match_init(query_words):
    patt = u'|'.join([re.escape(word) for word in query_words])
    def num_matches(sentence_string):   
        return len(re.findall(patt, sentence_string, re.I))
    return num_matches

def get_caps_rate(sentence):
    num_tokens = 0
    num_uppers = 0
    for token in sentence.tokens:
        if re.search(r'[A-Z]', token.token):
            num_uppers += 1
        num_tokens += 1
    if num_tokens > 0:
        return num_uppers / float(num_tokens)
    else:
        return 0

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

    parser.add_argument('--word-freqs',
                        help=u'Location of word count dir',
                        type=unicode, required=True)

    parser.add_argument('--doc-freqs',
                        help=u'Location of doc count dir',
                        type=unicode, required=True)

    parser.add_argument('--domain-lm-port',
                        help=u'domain lm port number',
                        type=str, required=True)

    parser.add_argument('--background-lm-ports',
                        help=u'background lm port number (3gram, 4gram, 5gram)',
                        type=str, nargs=3, required=True)

    args = parser.parse_args()
    event_file = args.event_file
    rc_dir = args.rel_chunks_dir
    event_title = args.event_title
    ofile = args.output_file
    dm_port = args.domain_lm_port
    bg_ports = args.background_lm_ports 
    wc_dir = args.word_freqs
    dc_dir = args.doc_freqs
    
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

    if not os.path.exists(wc_dir) or not os.path.isdir(wc_dir):
        sys.stderr.write((u'--word-freqs argument {} either does not' \
                          + u' exist or is not a directory!\n').format(wc_dir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(dc_dir) or not os.path.isdir(dc_dir):
        sys.stderr.write((u'--doc-freqs argument {} either does not' \
                          + u' exist or is not a directory!\n').format(dc_dir))
        sys.stderr.flush()
        sys.exit()

    return (event_file, rc_dir, event_title, ofile,
            (dm_port, bg_ports), (wc_dir, dc_dir))

if __name__ == '__main__':
    main()
