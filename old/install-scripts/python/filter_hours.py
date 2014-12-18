import argparse
import sys
import os
from cuttsum.event import read_events_xml
from cuttsum.util import gen_dates
from datetime import datetime, timedelta
import re

def main():

    event_xml, filter_list, hour_list = parse_args()
    
    events = read_events_xml(event_xml)
    valid_hours = set()

    for event in events:
        start_dt = event.start - timedelta(hours=5)
        end_dt = event.end
        for dth in gen_dates(start_dt, end_dt):
            valid_hours.add(dth)
    with open(hour_list, u'r') as f, open(filter_list, u'w') as o:
        for line in f:
            path = line.strip()
            dth, fname = path.split('/')
            if dth in valid_hours and re.search(r'news', fname, re.I):
                o.write(path)
                o.write('\n')
                o.flush()
                

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--events-xml',
                        help=u'Events xml file.',
                        type=unicode, required=True)

    parser.add_argument('-f', '--filter-list',
                        help=u'Chunk directory.',
                        type=unicode, required=True)

    parser.add_argument('-l', '--hour-list',
                        help=u'List of valid date hours.',
                        type=unicode, required=True)

    args = parser.parse_args()
    event_xml = args.events_xml
    filter_list = args.filter_list
    hour_list = args.hour_list

    if not os.path.exists(event_xml) or os.path.isdir(event_xml):
        sys.stderr.write((u'--events-xml argument {} either does not exist' \
                          + u' or is a directory!\n').format(event_xml))
        sys.stderr.flush()
        sys.exit()

    flist_dir = os.path.dirname(filter_list)
    if not os.path.exists(flist_dir):
        os.makedirs(flist_dir)

    if not os.path.exists(hour_list) or os.path.isdir(hour_list):
        sys.stderr.write((u'--hour-list argument {} either does not exist' \
                          + u' or is a directory!\n').format(hour_list))
        sys.stderr.flush()
        sys.exit()

    return event_xml, filter_list, hour_list

if __name__ == '__main__':
    main()

