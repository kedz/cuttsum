import streamcorpus as sc
import cuttsum.judgements
import cuttsum.events
import cuttsum.corpora
from cuttsum.misc import stringify_streamcorpus_sentence
import pandas as pd

event = cuttsum.events.get_events(by_query_ids=["TS13.1"])[0]
corpus = cuttsum.corpora.EnglishAndUnknown2013()

paths = ["/scratch/t-chkedz/trec-data/articles/gold/2012_buenos_aires_rail_disaster/2012-02-22-12.sc.gz",
         "/scratch/t-chkedz/trec-data/articles/gold/2012_buenos_aires_rail_disaster/2012-02-22-13.sc.gz",
         "/scratch/t-chkedz/trec-data/articles/gold/2012_buenos_aires_rail_disaster/2012-02-22-14.sc.gz",
         "/scratch/t-chkedz/trec-data/articles/gold/2012_buenos_aires_rail_disaster/2012-02-22-15.sc.gz",
         ]

matches = cuttsum.judgements.get_merged_dataframe()

#print matches[matches["update id"].apply(lambda x: x.startswith(example_id))]

def si_to_df(si):
    sents = []
    for s, sent in enumerate(si.body.sentences["lingpipe"]):
        sents.append({
            "sent id": s, 
            "sent text": stringify_streamcorpus_sentence(sent).decode("utf-8"),
            "doc id": si.stream_id,
            "update id": si.stream_id + "-" + str(s),
            })
    return pd.DataFrame(sents)


dfs = []
for path in paths:
    with sc.Chunk(path=path, mode="rb", message=corpus.sc_msg()) as chunk:

        for si in chunk:
            df = si_to_df(si)
            df["nuggets"] = df["update id"].apply(
                lambda x: set(matches[matches["update id"] == x]["nugget id"].tolist()))
            dfs.append(df)
df = pd.concat(dfs)
df = df[df["nuggets"].apply(len) > 0]

from sumpy.system import AverageFeatureRankerBase
from sumpy.annotators import LedeMixin, CentroidMixin
from sumpy.annotators import WordTokenizerMixin

class SubmodularMixin(WordTokenizerMixin):
    def build(self):
        if not hasattr(self, "k"):
            self.k = 5
        assert self.k > 0




    def process(self, input_df, ndarray_data):
        
        input_size = len(input_df)
        S = []
        N = set()

        n_of_e = input_df["nuggets"].tolist()
        V_min_S = [i for i in xrange(input_size)]
        f_of_S = 0        

        for i in xrange(self.k):
            arg_max = None
            gain_max = 0
            for pos, elem in enumerate(V_min_S):
                #print "elem", elem
                #print "S", S
                #print "V_min_S", V_min_S
                #print "n(e) =", n_of_e[elem]
                n_of_S_U_e = N.union(n_of_e[elem])
                #print "S U {e}", S + [elem]
                #print "n(S U {e})", n_of_S_U_e
                gain = len(n_of_S_U_e) - f_of_S
                #print "gain", gain
                #print
                if gain > gain_max: 
                    arg_max = pos
                    gain_max = gain

            if arg_max is not None:
                S = S + [V_min_S[arg_max]]
                N = N.union(n_of_e[V_min_S[arg_max]])
                f_of_S = len(N)
                
                print "ARG MAX", V_min_S[arg_max]
                print "S", S
                print "N", N
                print "f(S)", f_of_S
                print
                
                del V_min_S[arg_max]


        input_df.ix[S, "f:monotone-submod"] = 1        
        input_df.ix[V_min_S, "f:monotone-submod"] = 0        


        return input_df, ndarray_data


    def requires(self):
        return ["words"]
    
    def ndarray_requires(self):
        return []

    def returns(self):
        return ["f:montone-submod"]

    def ndarray_returns(self):
        return []

    def name(self):
        return "MonotoneSubmod" 


class SubmodularSummarizer(SubmodularMixin, AverageFeatureRankerBase):
    pass


import sumpy
def f_of_A(system, A, V_min_A, e, input_df, ndarray_data):
    return len(
        set([nugget for nuggets in input_df.ix[A, "nuggets"].tolist() for nugget in nuggets]))
system = sumpy.system.MonotoneSubmodularBasic(f_of_A=f_of_A)
summ = system.summarize(df).budget(size="all")
print summ
#print "\n".join(odf["sent text"].tolist())

