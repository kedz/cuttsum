import codecs
from geopy import distance
import os
import  streamcorpus as sc

def get_geocache(cache_path=None):

    cache = {}
    if cache_path is None:
        rdir = os.path.join(os.path.dirname(__file__), 'resources')
        cache_path = os.path.join(rdir, 'geocache.txt')
    with codecs.open(cache_path, 'r', 'utf-8') as f:
        for line in f:
            items = line.strip().split(u'\t')
            cache[items[0]] = (float(items[1]), float(items[2]))
    return cache

def compute_distance_stats(cache, locs1, locs2):
    min_distance = 20037 #half the circumference of earth 40075km
    max_distance = -20037
    tot_distance = 0
    distances = 0
    for loc1 in locs1:
        if loc1 not in cache:
            continue
        for loc2 in locs2:
            if loc2 not in cache:
                continue # circumference of earth in km
            d = distance.distance(cache[loc1], cache[loc2]).kilometers
            if d < min_distance:
                min_distance = d
            if d > max_distance:
                max_distance = d
            tot_distance += d
            distances += 1

    if distances > 0:
        avg_distance = tot_distance / float(distances)
    else:
        avg_distance = 0                

    return min_distance, max_distance, avg_distance

def get_loc_sequences(sentence):
    seqs = []
    buff = []
    for token in sentence.tokens:
        if is_loc(token):
            buff.append(token.token.decode('utf-8'))
        else:
            if len(buff) > 0:
                seqs.append(u' '.join(buff))
                buff = []
    if len(buff) > 0:
        seqs.append(u' '.join(buff))
        buff = []
    return seqs

def is_loc(token):
    netype = None
    if token.entity_type != 3 and token.entity_type != 4:
        netype = sc.get_entity_type(token)
    if netype == 'LOC':
        return True
    else:
        return False
