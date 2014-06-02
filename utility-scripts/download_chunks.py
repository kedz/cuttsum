import os
import argparse
from collections import OrderedDict
import re
from datetime import datetime, timedelta
import time
from calendar import timegm
from multiprocessing import Pool
import subprocess
from cuttsum.event import read_events_xml

import streamcorpus

def main():

    chunkurls_file, events_xml, data_dir = parse_args()
    events = read_events_xml(events_xml)

    for event in events:

        padded_start = event.start - timedelta(hours=24)
        print 'EVENT TITLE: {}'.format(event.title)
        print 'EVENT TYPE: {}'.format(event.type)
        print 'DATE RANGE: {} -- {}'.format(event.start, event.end)
        print 'DATE RANGE-24: {} -- {}'.format(padded_start, event.end)
        urls = read_urls(chunkurls_file, padded_start, event.end)
        print len(urls), 'IN RANGE'
       
        #ws = os.path.join(data_dir, event.title.replace(' ', '_')) 
        #if not os.path.exists(ws):
        #    os.makedirs(ws)
        ws = data_dir
        jobs = []
        for url in urls:
            prefix = url.split('/')[-2]
            path = os.path.join(ws, prefix)
            jobs.append((url, path))
            if not os.path.exists(path):
                os.makedirs(path)
        pool = Pool(4)
        njobs = len(jobs)
        results = pool.imap_unordered(download_url, jobs) 
        for i, result in enumerate(results, 1):
            print '({}/{}) Downloaded'.format(i, njobs), result


def download_url(args):
    url, ws = args
    url_toks = url.split('/')
    fpath = os.path.join(ws, url_toks[-1])
    xzpath = os.path.splitext(fpath)[0]
    if os.path.exists(xzpath):
        return xzpath
    with open(xzpath, 'wb') as f, open(os.devnull, "w") as fnull:
        subprocess.call(['wget', '-P', ws, '-t', '0', url],
                        stdout=fnull, stderr=fnull)
        subprocess.call(['gpg', '--decrypt', fpath],
                        stdout=f, stderr=fnull)
    os.remove(fpath)
    return xzpath


def read_urls(urls_file, start_date, end_date):
    good_urls = []
#    _zulu_timestamp_format = '%Y-%m-%dT%H:%M:%S.%f%Z'
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f.readlines()]
        for url in urls:
            url_toks = url.split('/')    
            #date_str = '-'.join(url_toks[-2].split('-')[:-1]) + \
            #    'T00:00:00.000000GMT'
            #then = time.strptime(date_str, _zulu_timestamp_format),

            date_str = '-'.join(url_toks[-2].split('-')[:-1]) + \
                'T00:00:00.000000Z'
            etstr = streamcorpus.make_stream_time(date_str).epoch_ticks
            urldate = datetime.utcfromtimestamp(etstr)
             #zulu_timestamp.replace('Z', 'GMT'),
              #           _zulu_timestamp_format.replace('Z', '%Z')
                          #           )
            #print then
            #timestamp = timegm(then[0])
            #print datetime.utcfromtimestamp(timestamp)
            #2000-01-01T12:34:00.000123Z
            #urldate = datetime.strptime(date_str, "%Y-%m-%d").date()
            if start_date <= urldate and urldate < end_date:
                good_urls.append(url)
    
    return good_urls

def read_eventdata(eventdata):

    with open(eventdata, 'r') as f:
        txt = ''.join(f.readlines())
        #print txt

        m = re.search(r'<title>(.*?)</title>', txt)
        title = m.groups()[0]
        m = re.search(r'<start>(.*?)</start>', txt)
        start = datetime.fromtimestamp(int(m.groups()[0]))
        m = re.search(r'<end>(.*?)</end>', txt)
        end = datetime.fromtimestamp(int(m.groups()[0]))
        m = re.search(r'<type>(.*?)</type>', txt)
        eventtype = m.groups()[0]
        m = re.search(r'<query>(.*?)</query>', txt)
        query = tuple(m.groups()[0].split(' '))
        return Event(title, eventtype, query, start, end)

class Event:
    def __init__(self, title, eventtype, query, start, end):
        self.title = title
        self.type = eventtype
        self.query = query
        self.start = start
        self.end = end

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chunk-list',
                        help=u'File with list of chunk urls.',
                        type=unicode, required=True)

    parser.add_argument('-e', '--event-file',
                        help=u'Event xml file.',
                        type=unicode, required=True)

    parser.add_argument('-w', '--workspace',
                        help=u'Workspace directory',
                        type=unicode, required=True)
    args = parser.parse_args()
    chunk_list = args.chunk_list
    event_file = args.event_file
    ws = args.workspace

    if not os.path.exists(chunk_list) or os.path.isdir(chunk_list):
        import sys
        sys.stderr.write((u'--chunk-list argument {} either does not exist' \
                          + u' or is a directory!\n').format(chunk_list))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(event_file) or os.path.isdir(event_file):
        import sys
        sys.stderr.write((u'--chunk-list argument {} either does not exist' \
                          + u' or is a directory!\n').format(event_file))
        sys.stderr.flush()
        sys.exit()

    if ws != '' and not os.path.exists(ws):
        os.makedirs(ws)

    return chunk_list, event_file, ws

if __name__ == '__main__':
    main()
