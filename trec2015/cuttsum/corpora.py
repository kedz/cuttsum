import streamcorpus as sc


def get_raw_corpus(event):
    if event.query_id.startswith(u"TS13"):
        return EnglishAndUnknown2013()
    elif event.query_id.startswith(u"TS14") \
            or event.query_id.startswith(u"TS15"):
        return SerifOnly2014()
    else:
        raise Exception("Bad query id {}".format(event.query_id))

class _KBAStreamCorpus(object):

    def year(self):
        return self.year_
        
    def version(self):
        return self.ver_

    def fs_name(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def annotator(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def sc_msg(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def is_subset(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_superset(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_sentences(self, si):
        if u'serif' in si.body.sentences:
            return si.body.sentences[u'serif']
        elif u'lingpipe' in si.body.sentences:
            return si.body.sentences[u'lingpipe']
        else:
            return []

class EnglishAndUnknown2013(_KBAStreamCorpus):
    """This KBA StreamCorpus contains only the English and unknown langauge
documents that werer processed by LingPipe."""

    def __init__(self):
        self.year_ = 2013
        self.ver_ = u'0.2.0'
        self.aws_url_ = u'http://s3.amazonaws.com/' + \
            u'aws-publicdatasets/trec/kba/' + \
            u'kba-streamcorpus-2013-v0_2_0-english-and-unknown-language/'

    def fs_name(self):
        return u'2013_english_and_unknown' 

    def annotator(self):
        return u'lingpipe'

    def sc_msg(self):
        return sc.StreamItem_v0_2_0            

    def is_subset(self):
        return False

    def get_superset(self):
        return self

class SerifOnly2014(_KBAStreamCorpus):
    """This KBA StreamCorpus contains only the English and unkown language
documents that were processed by the BBN Serif tool."""

    def __init__(self):
        self.year_ = 2014
        self.ver_ = u'0.3.0'
        self.aws_url_ = u'http://s3.amazonaws.com/' + \
            u'aws-publicdatasets/trec/kba/' + \
            u'kba-streamcorpus-2014-v0_3_0-serif-only/'

    def fs_name(self):
        return u'2014_serif_only' 

    def annotator(self):
        return u'serif'

    def sc_msg(self):
        return sc.StreamItem_v0_3_0            

    def is_subset(self):
        return False

    def get_superset(self):
        return self

class FilteredTS2014(_KBAStreamCorpus):
    """This KBA StreamCorpus contains only the English and unkown language
documents that were processed by the BBN Serif tool and was filtered with
the TREC TS 2014 event queries."""

    def __init__(self):
        self.year_ = 2014
        self.ver_ = u'0.3.0'
        self.aws_url_ = u'http://s3.amazonaws.com/' + \
            u'aws-publicdatasets/trec/ts/' + \
            u'streamcorpus-2014-v0_3_0-ts-filtered/'

    def fs_name(self):
        return u'2014_filtered_ts' 

    def annotator(self):
        return u'serif'

    def sc_msg(self):
        return sc.StreamItem_v0_3_0            

    def is_subset(self):
        return True

    def get_superset(self):
        return SerifOnly2014()

class FilteredTS2015(_KBAStreamCorpus):
    """This KBA StreamCorpus is a filtered version of the 2014 corpus 
made specifically for the TS 2015 events."""

    def __init__(self):
        self.year_ = 2015
        self.ver_ = u'1'
        self.aws_url_ = u'http://s3.amazonaws.com/' + \
            u'aws-publicdatasets/trec/ts/' + \
            u'streamcorpus-2015-v1-ts-filtered/'

    def fs_name(self):
        return u'Trec-TS-2015F' 

    def annotator(self):
        return u'serif'

    def sc_msg(self):
        return sc.StreamItem_v0_3_0            

    def is_subset(self):
        return True

    def get_superset(self):
        return SerifOnly2014()
