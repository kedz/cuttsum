import streamcorpus as sc
from .data import Resource, get_resource_manager
import os
import signal
import Queue
import gzip
import numpy as np
import pandas as pd
import marisa_trie
from sklearn.cluster import AffinityPropagation

def get_loc_sequences(doc):
    seqs = []
    for sentence in doc:
        buff = []
        for token in sentence:
            if token.ne == u'LOCATION':
                buff.append(token.surface)
            else:
                if len(buff) > 0:
                    seqs.append(u' '.join(buff))
                    buff = []
        if len(buff) > 0:
            seqs.append(u' '.join(buff))
            buff = []
    return seqs

def streamcorpus_is_loc(token):
    netype = None
    if token.entity_type != 3 and token.entity_type != 4:
        netype = sc.get_entity_type(token)
    if netype == 'LOC':
        return True
    else:
        return False


class GeoCacheResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'geo-cache')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)
        
        self.cache_fname_ = u'geo-cache.tsv'

    def get_tsv_path(self):
        return os.path.join(self.dir_, self.cache_fname_)

    def __unicode__(self):
        return u"cuttsum.geo.GeoCacheResource"

    def check_coverage(self, event, corpus, **kwargs):
        coverage = 0.0
        if os.path.exists(self.get_tsv_path()):
            coverage = 1.0
        
        return coverage

    def get(self, event, corpus, **kwargs):
        raise NotImplementedError(u'I do not know how to get this resource!')

    def dependencies(self):
        return tuple([])

class GeoClustersResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'geo-clusters')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def dependencies(self):
        return tuple([u'SentenceStringsResource', u'GeoCacheResource',])

    def __unicode__(self):
        return u"cuttsum.geo.GeoClustersResource"

    def get_tsv_path(self, event, hour):
        data_dir = os.path.join(self.dir_, event.fs_name())
        return os.path.join(data_dir, u'{}.tsv.gz'.format(
            hour.strftime(u'%Y-%m-%d-%H')))

    def check_coverage(self, event, corpus, **kwargs):
        
        n_hours = 0
        n_covered = 0

        strings = get_resource_manager(u'SentenceStringsResource')

        for hour in event.list_event_hours(preroll=1):
            n_hours += 1
            if os.path.exists(self.get_tsv_path(event, hour)):
                n_covered += 1

        if n_hours == 0:
            return 0
        else:
            return n_covered / float(n_hours)

    def get(self, event, corpus, overwrite=False, n_procs=1, 
            progress_bar=False, preroll=0, **kwargs):
        strings = get_resource_manager(u'SentenceStringsResource')
        data_dir = os.path.join(self.dir_, event.fs_name())
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        jobs = []
        for hour in event.list_event_hours(preroll=1):
            string_tsv_path = strings.get_tsv_path(event, hour)
            geo_tsv_path = self.get_tsv_path(event, hour)
            if os.path.exists(string_tsv_path):
                if overwrite is True or not os.path.exists(geo_tsv_path):
                    jobs.append((string_tsv_path, geo_tsv_path))    

        self.do_work(geo_worker_, jobs, n_procs, progress_bar,
                     event=event, corpus=corpus)

def geo_worker_(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
 
    geocache = get_resource_manager(u'GeoCacheResource')
    geoquery = GeoQuery(geocache.get_tsv_path())
    event = kwargs.get(u'event')

    while not job_queue.empty():
        try:
            string_tsv_path, geo_tsv_path = \
                job_queue.get(block=False)

            with gzip.open(string_tsv_path, u'r') as f:
                string_df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3, header=0)

            loc_strings = [loc_string for loc_string 
                           in string_df[u'locations'].tolist()
                           if not isinstance(loc_string, float)]

            coords = []
            
            for loc_string in loc_strings:
                for location in loc_string.split(','):
                    coord = geoquery.lookup_location(location)
                    if coord is not None:
                        coords.append(coord)

            centers = set() 
            if len(coords) > 0:
                coords = np.array(coords)
                D = -geoquery.compute_distances(coords[:,None], coords)
                ap = AffinityPropagation(affinity=u'precomputed')
                Y = ap.fit_predict(D)     
              
                if ap.cluster_centers_indices_ is not None: 
                    for center in ap.cluster_centers_indices_:
                        centers.add((coords[center][0], coords[center][1]))
            
                    centers = [{u'lat': lat, u'lng': lng} 
                               for lat, lng in centers]
                    centers_df = pd.DataFrame(centers, columns=[u'lat', u'lng'])
                
                    with gzip.open(geo_tsv_path, u'w') as f:
                        centers_df.to_csv(f, sep='\t', index=False, 
                                          index_label=False, na_rep='nan')  

            result_queue.put(None)
        except Queue.Empty:
            pass

    return True

class GeoQuery(object):
    def __init__(self, geo_cache_tsv_path):
        self.trie_ = self.load_(geo_cache_tsv_path)

    def lookup_locations(self, locations):
        return [self.lookup_location(loc) for loc in locations]

    def lookup_location(self, location):
        if isinstance(location, str):
            location = location.decode(u'utf-8')
        location = location.lower()
        r = self.trie_.get(location, None)
        if r is not None:
            return r[0]
        else:
            return None
        
    def compute_distances(self, pos1, pos2, r=3958.75):
        pos1 = pos1 * np.pi / 180
        pos2 = pos2 * np.pi / 180
        cos_lat1 = np.cos(pos1[..., 0])
        cos_lat2 = np.cos(pos2[..., 0])
        cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
        cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
        return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

    def load_(self, geo_cache_tsv_path):
        geo_data = {}
        with open(geo_cache_tsv_path, u'r') as f:
            for line in f:
                loc, lat, lng = line.strip().split('\t')
                loc = loc.decode(u'utf-8').lower()
                lat = float(lat)
                lng = float(lng)
                geo_data[loc] = (lat, lng)
        trie = marisa_trie.RecordTrie("<dd", geo_data.items())
        return trie
