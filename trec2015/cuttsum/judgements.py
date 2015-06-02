from datetime import datetime
from pkg_resources import resource_stream, resource_filename
import gzip
import pandas as pd
import os
#from pkg_resources import resource_filename
#def this_is_a_test():
#    print "High"

def convert_to_datetime(x):
    return datetime.utcfromtimestamp(int(x))

def get_2014_nuggets():

    nuggets_tsv = resource_filename(
        u'cuttsum', os.path.join(u'2014-data', u'nuggets.tsv.gz'))
    with gzip.open(nuggets_tsv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep='\t', quoting=3, header=0,
            converters={u'timestamp': convert_to_datetime},
            names=[u'query id', u'nugget id', u'timestamp',
                   u'important', u'length', 'text'])
    return df

def get_2013_nuggets():
    nuggets_tsv = resource_filename(
        u'cuttsum', os.path.join(u'2013-data', u'nuggets.tsv.gz'))
    with gzip.open(nuggets_tsv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep='\t', quoting=3, header=0,
            converters={u'timestamp': convert_to_datetime},
            names=[u'query id', u'nugget id', u'timestamp',
                   u'important', u'length', 'text'])
    return df


def get_2013_matches():
    matches_tsv = resource_filename(
            u'cuttsum', os.path.join(u'2013-data', u'matches.tsv.gz'))
    with gzip.open(matches_tsv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep='\t', quoting=3, header=0,
            dtype={u'match start': int, u'match end': int},
            names=[u'query id', u'update id', u'nugget id',
                   u'match start', u'match end', 'auto p'])
    return df

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

def get_mturk_matches():
    matches_tsv = resource_filename(
            u'cuttsum', os.path.join(u'2015-data', u'mturk-matches.tsv.gz'))
    with gzip.open(matches_tsv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep='\t', quoting=3, header=0,
            dtype={u'match start': int, u'match end': int},
            names=[u'query id', u'update id', u'nugget id',
                   u'match start', u'match end', 'auto p'])
    return df

def get_2013_updates():
    updates_tsv = resource_filename(
        u'cuttsum', os.path.join(u'2013-data', u'updates.tsv.gz'))
 
    with gzip.open(updates_tsv, u'r') as f:
        df = pd.io.parsers.read_csv(
            f, sep='\t', quoting=3, header=0,
            names=[u'query id', u'update id', u'document id',
                   u'sentence id', u'length', u'duplicate id', u'text'])

    return df

def get_merged_dataframe():
    matches = get_2013_matches()
    updates = get_2013_updates()
    nuggets = get_2013_nuggets()

    match_cols = (matches.columns.difference(updates.columns)).tolist()
    match_cols += [u"update id"]

    matches_updates = pd.merge(
        left=matches[match_cols], right=updates, on="update id")

    #nugget_cols = (nuggets.columns.difference(
    #    matches_updates.columns)).tolist()
    matches_updates.rename(columns={u"text": u"update text"}, inplace=True)
    nuggets.rename(
        columns={u"text": u"nugget text", 
                 u"length": u"nugget length"}, 
        inplace=True)
    nugget_cols = [u'nugget id', u'timestamp', u'important', u'nugget length', 
        u'nugget text']
   
    return pd.merge(left=nuggets[nugget_cols], right=matches_updates, on="nugget id")
    #print matches_updates.columns.tolist()
    #print nuggets.columns.tolist()
    #import sys
    #3sys.exit()
