import xml.etree.cElementTree as ET
from datetime import datetime
import re
from pkg_resources import resource_stream
import os

def get_2013_events(by_query_ids=None, by_title=None):
    event_xml = resource_stream(
        u'cuttsum', os.path.join(
            u'2013-data', u'trec2013-ts-topics-test.xml'))
    return get_events(event_xml, by_query_ids=by_query_ids, by_title=by_title)

def get_2014_events(by_query_ids=None, by_title=None):
    event_xml = resource_stream(
        u'cuttsum', os.path.join(
            u'2014-data', u'trec2014-ts-topics-test.xml'))
    return get_events(event_xml, by_query_ids=by_query_ids, by_title=by_title)

def get_events(event_xml, by_query_ids=None, by_title=None):
    if isinstance(by_query_ids, list):
        by_query_ids = set(by_query_ids)
    ts_events = []
    for event, elem in ET.iterparse(event_xml, events=('end',)):
        if elem.tag == 'event':
            query_id = u'TS14.{}'.format(elem.findtext('id'))
            title = elem.findtext('title')    
            if isinstance(title, str):
                title = title.decode(u'utf-8')
            event_type = elem.findtext('type')
            if isinstance(event_type, str):
                event_type = event_type.decode(u'utf-8')
            query = elem.findtext('query')
            if isinstance(query, str):
                query = query.decode(u'utf-8')
            query = tuple(query.split(u' '))
        
            start = datetime.utcfromtimestamp(int(elem.findtext('start')))
            end = datetime.utcfromtimestamp(int(elem.findtext('end')))

            if by_query_ids is not None:
                if query_id in by_query_ids:
                    ts_events.append(
                        Event(query_id, title, event_type, query, start, end))
            elif by_title is not None:
                if by_title == title:
                    ts_events.append(
                        Event(query_id, title, event_type, query, start, end))
            else:
                ts_events.append(
                    Event(query_id, title, event_type, query, start, end))

    return ts_events

class Event:
    def __init__(self, query_id, title, type, query, start, end):
        self.query_id = query_id
        self.title = title
        self.type = type
        self.query = query
        self.start = start
        self.end = end

    def fs_name(self):
        title = re.sub(r'[(){}\[\]]', r'', self.title)
        title = re.sub(r'\s+', r'_', title)
        title = title.lower()
        return title

    def regex_pattern(self):
        patt = u' '.join(self.query)
        patt = re.sub(r'[\d(){}\[\]]', r'', patt)
        patt = re.sub(r'\s+', r'|', patt)
        return re.compile(patt)

    def __unicode__(self):
        return u"{} {}: ``{}'' {}--{} {}".format(
            self.query_id, self.type, self.title, self.start, self.end,
            self.query)
        
    def __str__(self):
        return unicode(self).encode(u'utf-8')

    def duration_hours(self):
        duration = self.end - self.start
        days, seconds = duration.days, duration.seconds
        hours = days * 24 + seconds // 3600
        return hours

def main(args):


    if args.relevance_report is True:
        import pandas as pd
        import cuttsum.data as data
        report = []
        for event in get_2013_events():
            chunks = data.relevant_chunks(event)
            n_chunks = len(chunks)
            duration = event.end - event.start
            days, seconds = duration.days, duration.seconds
            hours = days * 24 + seconds // 3600
            chunk_size = data.relevant_chunk_size(event)
            report.append({'query id': event.query_id,
                           'duration hours': hours,
                           'possibly relevant hours': n_chunks,
                           "chunk size (mb)": chunk_size})
        for event in get_2014_events():
            chunks = data.relevant_chunks(event)
            n_chunks = len(chunks)
            duration = event.end - event.start
            days, seconds = duration.days, duration.seconds
            hours = days * 24 + seconds // 3600
            chunk_size = data.relevant_chunk_size(event)
            report.append({'query id': event.query_id, 
                           'duration hours': hours,
                           'possibly relevant hours': n_chunks, 
                           "chunk size (mb)": chunk_size })

        print pd.DataFrame(report)


if __name__ == u'__main__':
    import argparse
    parser = argparse.ArgumentParser()
#    parser.add_argument(u'-q', u'--query-ids', nargs=u'+',
#                        help=u'Event query ids, e.g. TS14.11',
#                        default=None, required=False)

#    parser.add_argument(u'-p', u'--n-procs', type=int,
#                        help=u'Number of processes to run',
#                        default=1, required=False)

    parser.add_argument(u'--relevance-report', action=u'store_true',
                        help=u'Print relevance report')

    args = parser.parse_args()
    main(args)
