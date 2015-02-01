import xml.etree.cElementTree as ET
from datetime import datetime, timedelta
import re
from pkg_resources import resource_stream
import os

def get_2013_events(by_query_ids=None, by_event_types=None):
    event_xml = resource_stream(
        u'cuttsum', os.path.join(
            u'2013-data', u'trec2013-ts-topics-test.xml'))
    return get_events(event_xml, by_query_ids=by_query_ids,
                      by_event_types=by_event_types, prefix='TS13.')

def get_2014_events(by_query_ids=None, by_event_types=None):
    event_xml = resource_stream(
        u'cuttsum', os.path.join(
            u'2014-data', u'trec2014-ts-topics-test.xml'))
    return get_events(event_xml, by_query_ids=by_query_ids, 
                      by_event_types=by_event_types, prefix='TS14.')

def get_events(event_xml, by_query_ids=None, by_event_types=None, prefix=''):
    if isinstance(by_query_ids, list):
        by_query_ids = set(by_query_ids)
    if isinstance(by_event_types, list):
        by_event_types = set(by_event_types)
    ts_events = []
    for event, elem in ET.iterparse(event_xml, events=('end',)):
        if elem.tag == 'event':
            query_id = u'{}{}'.format(prefix, elem.findtext('id'))
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
            wiki = elem.findtext('description')
        
            start = datetime.utcfromtimestamp(int(elem.findtext('start')))
            end = datetime.utcfromtimestamp(int(elem.findtext('end')))

            if by_query_ids is not None:
                if query_id in by_query_ids:
                    ts_events.append(
                        Event(query_id, title, event_type, query, start, end, wiki))
            elif by_event_types is not None:
                if event_type in by_event_types:
                    ts_events.append(
                        Event(query_id, title, event_type, query, start, end, wiki))
            else:
                ts_events.append(
                    Event(query_id, title, event_type, query, start, end, wiki))

    return ts_events

class Event:
    def __init__(self, query_id, title, type, query, start, end, wiki):
        self.query_id = query_id
        self.query_num = int(query_id.split('.')[1])
        self.title = title
        self.type = type
        self.query = query
        self.start = start
        self.end = end
        self.wiki = wiki

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
 
    def list_event_hours(self, preroll=0):
        start_dt = self.start.replace(minute=0, second=0) \
            - timedelta(hours=preroll)
        end_dt = self.end.replace(minute=0, second=0)
        current_dt = start_dt
        hours = []
        while current_dt <= end_dt:
            hours.append(current_dt)
            current_dt += timedelta(hours=1)
        return hours
