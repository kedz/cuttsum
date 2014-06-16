import re
from datetime import datetime
import xml.etree.cElementTree as ET

def read_events_xml(eventxml):
    ts_events = []
    for event, elem in ET.iterparse(eventxml, events=('end',)):
        if elem.tag == 'event':
            query_id = int(elem.findtext('id'))
            title = elem.findtext('title')    
            event_type = elem.findtext('type')
            query = elem.findtext('query').split(' ')
            start = datetime.utcfromtimestamp(int(elem.findtext('start')))
            end = datetime.utcfromtimestamp(int(elem.findtext('end')))
            ts_events.append(Event(query_id, title, event_type, 
                                   query, start, end))
    return ts_events

class Event:
    def __init__(self, query_id, title, eventtype, query, start, end):
        self.query_id = query_id
        self.title = title
        self.type = eventtype
        self.query = query
        self.start = start
        self.end = end
