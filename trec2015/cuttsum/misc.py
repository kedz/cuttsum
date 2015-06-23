import sys
from functools import reduce
import re
import pandas as pd

class ProgressBar:
    def __init__(self, max_jobs):
        self.max_jobs_ = max_jobs
        self.n_job_ = 0
        self.term_width_ = 70
        self.bin_size_ = max_jobs / float(self.term_width_)
        self.current_state_ = 0
    def update(self):
        if sys.stdout.isatty():
            self._update_term()
        else:
            self._update_log()

    def _update_term(self):
        self.n_job_ += 1
        if self.n_job_ == self.max_jobs_:
            sys.stdout.write(' ' * 79 + '\r')
            sys.stdout.flush()
        else:
            bins = int(self.n_job_ / self.bin_size_)
            bins = min(bins, self.term_width_)

            if bins > 0:
                bar = '\r [' + '=' * (bins - 1) + '>' + \
                    ' ' * (self.term_width_ - bins) + ']\r'
            else:
                bar = '\r [' + ' ' * self.term_width_ + ']\r'

            per = ' {:6.3f}% '.format(self.n_job_ * 100. / self.max_jobs_)
            bar = bar[:36] + per + bar[42:]

            sys.stdout.write(bar)
            sys.stdout.flush()

    def _update_log(self):
        self.n_job_ += 1
        if self.n_job_ == self.max_jobs_:
            sys.stdout.write('\n')
            sys.stdout.flush()
        else:
            per_complete = self.n_job_ * 100. / self.max_jobs_
            state = round(per_complete / 5)
            #print self.n_job_, state
            if state != self.current_state_:
                self.current_state_ = state
                if state % 4 == 0:             
                    sys.stdout.write('|')
                else:
                    sys.stdout.write('.')
                sys.stdout.flush()



    def clear(self):
        sys.stdout.write('\r')
        sys.stdout.write(' ' * 79)
        sys.stdout.write('\r')
        sys.stdout.flush()


def toposort(data):
    """Dependencies are expressed as a dictionary whose keys are items
and whose values are a set of dependent items. Output is a list of
sets in topological order. The first set consists of items with no
dependences, each subsequent set consists of items that depend upon
items in the preceeding sets.

>>> print '\\n'.join(repr(sorted(x)) for x in toposort2({
...     2: set([11]),
...     9: set([11,8]),
...     10: set([11,3]),
...     11: set([7,5]),
...     8: set([7,3]),
...     }) )
[3, 5, 7]
[8, 11]
[2, 9, 10]

"""


    # Ignore self dependencies.
    for k, v in data.items():
        v.discard(k)
    # Find all items that don't depend on anything.
    extra_items_in_deps = \
        reduce(set.union, data.itervalues()) - set(data.iterkeys())
    # Add empty dependences where needed
    data.update({item:set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.iteritems() if not dep)
        if not ordered:
            break
        yield ordered
        data = {item: (dep - ordered)
                for item, dep in data.iteritems()
                    if item not in ordered}
    assert not data, \
        "Cyclic dependencies exist among these items:\n{}".format(
            '\n'.join(repr(x) for x in data.iteritems()))

def stringify_streamcorpus_sentence(sentence):
    return ' '.join(token.token for token in sentence.tokens)

def stringify_corenlp_doc(doc):
    return ' '.join(stringify_corenlp_sentence(sent) for sent in doc)

def stringify_corenlp_sentence(sentence):
    normalized_tokens = []
    for token in sentence:
        if token.ne == 'O':
            words = token.lem.split(u'_')
            for word in words:
                if word != u'':
                    normalized_tokens.append(word.lower())
        else: 
            normalized_tokens.append(
                u'__{}__'.format(token.ne.lower()))
    return (u' '.join(normalized_tokens)).encode(u'utf-8')

def passes_simple_filter(scstring, doclen):
    words = len(re.findall(r'\b[^\W\d_]+\b', scstring))
    socs = len(re.findall(
        r'Digg|del\.icio\.us|Facebook|Kwoff|Myspace',
        scstring))  
    langs = len(re.findall(
        r'Flash|JavaScript|CSS', scstring, re.I))

    if words > 9 and doclen < 200 \
        and socs < 2 and langs < 2:
        
        return True
    else:
        return False
    
def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

def si2df(si, extractor=None):

    sents = []
    if extractor is None or extractor == "gold":
    
        for s, sent in enumerate(si.body.sentences["lingpipe"]):
            sents.append({
                "timestamp": int(si.stream_id.split("-")[0]),
                "sent id": s,
                "sent text": stringify_streamcorpus_sentence(
                    sent).decode("utf-8"),
                "doc id": si.stream_id,
                "words": [token.token for token in sent.tokens],
                "update id": si.stream_id + "-" + str(s),
                })
    elif extractor == "goose":
        
        for s, sent in enumerate(si.body.sentences["goose"]):
            if len(sent.tokens) == 0:
                continue
            sents.append({
                "timestamp": int(si.stream_id.split("-")[0]),
                "sent id": s,
                "sent text": stringify_streamcorpus_sentence(
                    sent).decode("utf-8"),
                "stream id": si.stream_id,
                "words": [token.token for token in sent.tokens],
                "update id": si.stream_id + "-" + str(s),
                })
    return pd.DataFrame(sents)

def english_stopwords():
    from pkg_resources import resource_stream
    import os
    stopwords_stream = resource_stream(
        u'cuttsum', os.path.join(u"misc-data", u"stopwords.en.txt"))
    return set(map(lambda x: x.strip(), stopwords_stream.readlines()))

def event2lm_name(event):
    if event.type == "accident":
        return "accidents-lm"
