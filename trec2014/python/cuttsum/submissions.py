import os
import gzip
from pkg_resources import resource_filename
import pandas as pd
from datetime import datetime

class Update(object):
    def __init__(self, system_id, run_id, query_id, document_id, sentence_id,
                 update_datetime, confidence):
        self.system_id = system_id
        self.run_id = run_id
        self.query_id = query_id
        self.document_id = document_id
        self.sentence_id = sentence_id
        self.update_datetime = update_datetime
        self.confidence = confidence

__dt_cvrt = lambda x: datetime.utcfromtimestamp(int(x))

def get_2014_system_updates():
    updates_ssv = resource_filename(
        u'cuttsum', os.path.join(u'2014-data', u'cunlp-updates.ssv.gz'))
    with gzip.open(updates_ssv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep=' ', quoting=3, header=None,
            converters={u'query id': lambda x: 'TS14.{}'.format(x),
                        u'update datetime': __dt_cvrt},
            names=[u'query id', u'system id', u'run id', u'document id',
                   u'sentence id', u'update datetime', 'confidence'])
        
    df.insert(3, u'update id',
        df[u'document id'].map(str) + '-' + df[u'sentence id'].map(str))
    df.insert(5, 'document timestamp', df[u'document id'].apply(
        lambda x: __dt_cvrt(x.split('-')[0])))
    return df

def get_2014_sampled_updates():
    updates_tsv = resource_filename(
        u'cuttsum', os.path.join(u'2014-data', u'updates_sampled.tsv.gz'))
 
    with gzip.open(updates_tsv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep='\t', quoting=3, header=0,
            #converters={u'query id': lambda x: 'TS14.{}'.format(x),
           #             u'update datetime': __dt_cvrt},
            names=[u'query id', u'update id', u'document id',
                   u'sentence id', u'length', u'duplicate id', u'text'])

    return df

def get_2014_sampled_updates_extended():
    updates_tsv = resource_filename(
        u'cuttsum', os.path.join(u'2014-data', u'updates_sampled.extended.tsv.gz'))
 
    with gzip.open(updates_tsv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep='\t', quoting=3, header=None,
            #converters={u'query id': lambda x: 'TS14.{}'.format(x),
           #             u'update datetime': __dt_cvrt},
            names=[u'query id', u'update id', u'document id',
                   u'sentence id', u'length', u'duplicate id', u'text'])

    return df

def get_2014_sampled_updates_levenshtein():
    updates_tsv = resource_filename(
        u'cuttsum', os.path.join(u'2014-data', u'updates_sampled.levenshtein.tsv.gz'))
    with gzip.open(updates_tsv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep='\t', quoting=3, header=None,
            #converters={u'query id': lambda x: 'TS14.{}'.format(x),
           #             u'update datetime': __dt_cvrt},
            names=[u'query id', u'update id', u'document id',
                   u'sentence id', u'length', u'duplicate id', u'text'])

    return df

