import codecs
from datetime import datetime
from pkg_resources import resource_stream, resource_filename
import os
import gzip
import pandas as pd

class Nugget(object):
    def __init__(self, query_id, nugget_id, timestamp, importance,
                 length, text):
        self.query_id = query_id
        self.nugget_id = nugget_id
        self.timestamp = timestamp
        self.importance = importance
        self.length = length
        self.text = text

        epoch = datetime.utcfromtimestamp(0)
        delta = timestamp - epoch
        self.epoch = int(delta.total_seconds())

    def __unicode__(self):
        return u'{} {} {}: {}'.format(
            self.nugget_id, self.importance, self.timestamp, self.text)
    
    def __str__(self):
        return unicode(self).encode(u'utf-8')

__dt_cvrt = lambda x: datetime.utcfromtimestamp(int(x))

def get_2014_nuggets():

    nuggets_tsv = resource_filename(
        u'cuttsum', os.path.join(u'2014-data', u'nuggets.tsv.gz'))
    with gzip.open(nuggets_tsv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep='\t', quoting=3, header=0,
            converters={u'timestamp': lambda x: __dt_cvrt},
            names=[u'query id', u'nugget id', u'timestamp',
                   u'important', u'length', 'text'])
    return df

    #nuggets_tsv = u'../TS14-data/nuggets.tsv'
#    nuggets = []
#    with codecs.open(nuggets_tsv, u'r', u'utf-8') as f:
#        f.readline()
#        for line in f:
#            items = line.strip().split('\t')
#            query_id = items[0]
#            nugget_id = items[1]
#            timestamp = datetime.utcfromtimestamp(int(items[2]))
#            importance = int(items[3])
#            length = int(items[4])
#            text = items[5]
#            if by_query_id is None or by_query_id == query_id:
#                nuggets.append(
#                    Nugget(query_id, nugget_id, timestamp,
#                           importance, length, text))
#    return nuggets

class Match(object):
    def __init__(self, query_id, update_id, nugget_id,
                 match_start, match_end, auto_p):
        self.query_id = query_id
        self.update_id = update_id
        self.nugget_id = nugget_id
        self.match_start = match_start
        self.match_end = match_end
        self.auto_p = auto_p

        self.epoch = int(update_id.split(u'-')[0])
        self.timestamp = datetime.utcfromtimestamp(self.epoch)

    def __str__(self):
       return unicode(self).encode(u'utf-8')

    def __unicode__(self):
        return (u'Query ID: {} Update ID {}: ' \
            + u'Nugget ID: {} Start-Stop {}-{} Auto-p {}').format(
            self.query_id, self.update_id, self.nugget_id, self.match_start,
            self.match_end, self.auto_p)

def get_2014_matches():
    matches_tsv = resource_filename(
            u'cuttsum', os.path.join(u'2014-data', u'matches.tsv.gz'))
    with gzip.open(matches_tsv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep='\t', quoting=3, header=0,
            dtype={u'match start': int, u'match end': int},
            names=[u'query id', u'update id', u'nugget id',
                   u'match start', u'match end', 'auto p'])
    return df

#def get_2014_matches(by_query_id=None, by_nugget_id=None):
#    matches_tsv = u'../TS14-data/matches.tsv'
#    matches = []
#    with codecs.open(matches_tsv, u'r', u'utf-8') as f:
#        f.readline()
#        for line in f:
#            items = line.strip().split(u'\t')
#            query_id = items[0]
#            update_id = items[1]
#            nugget_id = items[2]
#            match_start = int(items[3])
#            match_end = int(items[4])
#            auto_p = float(items[5])
#            if by_query_id is not None:
#                if by_query_id == query_id:
#                    m = Match(query_id, update_id, nugget_id,
#                              match_start, match_end, auto_p)
#                    matches.append(m)
#            elif by_nugget_id is not None:
#                if by_nugget_id == nugget_id:
#                    m = Match(query_id, update_id, nugget_id,
#                              match_start, match_end, auto_p)
#                    matches.append(m)
#            else:
#                m = Match(query_id, update_id, nugget_id,
#                          match_start, match_end, auto_p)
#                matches.append(m)
#
#    return matches
