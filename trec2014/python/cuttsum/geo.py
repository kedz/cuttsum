import streamcorpus as sc

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
        return u"cuttsum.lm.GeoCacheResource"

    def check_coverage(self, event, corpus, **kwargs):
        coverage = 0.0
        if os.path.exists(self.get_tsv_path()):
            coverage = 1.0
        
        return coverage

    def get(self, event, corpus, **kwargs):
        raise NotImplementedError(u'I do not know how to get this resource!')

    def dependencies(self):
        return tuple([])

