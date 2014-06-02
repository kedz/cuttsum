import re
from datetime import datetime
import xml.etree.cElementTree as ET

def read_events_xml(eventxml):
    ts_events = []
    for event, elem in ET.iterparse(eventxml, events=('end',)):
        if elem.tag == 'event':
            title = elem.findtext('title')    
            event_type = elem.findtext('type')
            query = elem.findtext('query').split(' ')
            start = datetime.utcfromtimestamp(int(elem.findtext('start')))
            end = datetime.utcfromtimestamp(int(elem.findtext('end')))
            ts_events.append(Event(title, event_type, query, start, end))
    return ts_events

def read_eventdata(eventdata):

    with open(eventdata, 'r') as f:
        txt = ''.join(f.readlines())

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
