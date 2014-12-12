import xml.etree.cElementTree as ET
from datetime import datetime
import re
from pkg_resources import resource_stream
import os

def get_2013_events(by_query_id=None, by_title=None):
    event_xml = resource_stream(
        u'cuttsum', os.path.join(
            u'2013-data', u'trec2013-ts-topics-test.xml'))
    return get_events(event_xml, by_query_id=by_query_id, by_title=by_title)

def get_2014_events(by_query_id=None, by_title=None):
    event_xml = resource_stream(
        u'cuttsum', os.path.join(
            u'2014-data', u'trec2014-ts-topics-test.xml'))
    return get_events(event_xml, by_query_id=by_query_id, by_title=by_title)

def get_events(event_xml, by_query_id=None, by_title=None):
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

            if by_query_id is not None:
                if by_query_id == query_id:
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
