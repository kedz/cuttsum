import os
import sys
import argparse
from cuttsum.event import read_events_xml
from cuttsum.nuggets import read_nuggets_tsv
from cuttsum.util import gen_dates
from nltk.tokenize import RegexpTokenizer
import codecs

def main():

    event_file, event_title, nuggets_tsv, odir = parse_args()
    event = load_event(event_title, event_file)
    nuggets = read_nuggets_tsv(nuggets_tsv, filter_query_id=event.query_id)
    hours = [dth for dth in gen_dates(event.start, event.end)]

    updates_file = os.path.join(odir, u'updates.txt')
    write_updates(nuggets, updates_file)

    sum_sents = []
    for hour in hours: 
        while len(nuggets) > 0:
            if nuggets[0].timestamp.strftime(u'%Y-%m-%d-%H') <= hour:
                sum_sents.append(nuggets[0].text)
                nuggets.pop(0)
            else:
                break

        if len(sum_sents) > 0:
            ofile = os.path.join(odir, u'{}.txt'.format(hour))
            write_summary(sum_sents, ofile)

    if len(nuggets) > 0:
        sum_sents.extend([nugget.text for nugget in nuggets])

    ofile = os.path.join(odir, u'final.txt')
    write_summary(sum_sents, ofile)
    

def write_updates(nuggets, ofile):

    with codecs.open(ofile, u'w', u'utf-8') as f:
        for nugget in nuggets:    
            t = nugget.timestamp.strftime(u'%Y-%m-%d-%H-%M-%S')
            f.write(u'{}\t{}\n'.format(t, nugget.text)) 
            f.flush()

def write_summary(texts, ofile):
    word_tokenizer = RegexpTokenizer(r'\w+')
    with codecs.open(ofile, u'w', u'utf-8') as f:
        for text in texts:
            f.write(u' '.join([w.lower() for w in word_tokenizer.tokenize(text)]))
            f.write(u'\n')
            f.flush()

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

    parser.add_argument('-n', '--nuggets-tsv',
                        help=u'Nuggets tsv file',
                        type=str, required=True)

    parser.add_argument('-t', '--event-title',
                        help=u'Event title',
                        type=unicode, required=True)

    parser.add_argument('-o', '--output-dir',
                        help=u'Location to write sims',
                        type=unicode, required=True)

    args = parser.parse_args()
    event_file = args.event_file
    event_title = args.event_title
    nuggets_tsv = args.nuggets_tsv
    odir = args.output_dir
    
    if odir != u'' and not os.path.exists(odir):
        os.makedirs(odir)

    if not os.path.exists(event_file) or os.path.isdir(event_file):
        sys.stderr.write((u'--event-file argument {} either does not exist' \
                          + u' or is a directory!\n').format(event_file))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(nuggets_tsv) or os.path.isdir(nuggets_tsv):
        sys.stderr.write((u'--nuggets-tsv argument {} either does not' \
                          + u' exist or is a directory!\n').format(
                            nuggets_tsv))
        sys.stderr.flush()
        sys.exit()

    return (event_file, event_title, nuggets_tsv, odir)

if __name__ == '__main__':
    main()
