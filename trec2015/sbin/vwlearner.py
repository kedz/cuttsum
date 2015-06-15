import cuttsum.events
import cuttsum.corpora
from cuttsum.pipeline import DedupedArticlesResource
import pyvw
import numpy as np



class UpdateSummarizer(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.AUTO_HAMMING_LOSS | sch.IS_LDF | sch.AUTO_CONDITION_FEATURES )

    def make_example(self, sentence, ncache):
        nuggets = sentence["nuggets"]
        ex = self.vw.example(
            {"a": [nid for nid in nuggets] if len(nuggets) > 0 else ["none"],
             "b": [nid + "_in_cache" for nid in ncache]},
            labelType=self.vw.lCostSensitive)
        return ex

    def _run(self, doc_iter):

        ncache = set()
        output = []
        n = 0

        for doc_orig in doc_iter:

            doc = list(doc_orig)

            while len(doc) > 0:
                n += 1
                print ncache
                examples = [self.make_example(sent, ncache) for sent in doc]
                gain = map(lambda x: len(x["nuggets"].difference(ncache)), doc)
                print gain
                examples.append(
                    self.vw.example(
                        {"c": ["all_clear" if np.sum(gain) == 0 else "stay"],},
                        labelType=self.vw.lCostSensitive))
                
                oracle = np.argmax(gain)
                oracle_gain = gain[oracle] 
                if oracle_gain == 0:
                    oracle = len(doc)
                pred = self.sch.predict(
                    examples=examples,
                    my_tag=n,
                    oracle=oracle,
                    condition=[ (n-1, "p"), ])
                output.append(pred)
                
                print "NEXT DOC CODE=", len(doc)
                print "PREDICTING", pred
                print "ORACLE", oracle

                if pred < len(doc):
                    ncache.update(doc[pred]["nuggets"])
                    print "Adding sent", doc[pred]["sent text"]
                    del doc[pred]
                                        
                if pred == len(doc):
                    break
                if oracle == len(doc):
                    break
   
        return output

event = cuttsum.events.get_events()[0]
corpus = cuttsum.corpora.EnglishAndUnknown2013()
extractor = "goose" 
res = DedupedArticlesResource()

def dfwrap(dfiter):
    for df in dfiter:
        yield df.to_dict(orient="records")

def new_df_iter():

    dfiter = res.dataframe_iter(
        event, corpus, extractor, include_matches="soft")
    return dfwrap(dfiter)

        

vw = pyvw.vw("--search 0 --csoaa_ldf m --search_task hook --ring_size 1024 --quiet --invert_hash mymodel.mdl --readable_model mymodel2.mdl")
task = vw.init_search_task(UpdateSummarizer)
my_iter = new_df_iter()
X =[[next(my_iter), next(my_iter), next(my_iter), next(my_iter), next(my_iter)]]
for x in range(4):
    task.learn(X)
sequence = task.predict(X[0])
print sequence


feats = ['VMTS13.01.064', 'VMTS13.01.058', 'VMTS13.01.056', 'VMTS13.01.055', 'VMTS13.01.054', 'VMTS13.01.052', 'VMTS13.01.051', 'VMTS13.01.050', 'VMTS13.01.062', 'VMTS13.01.070', 'VMTS13.01.060', 'VMTS13.01.077', 'VMTS13.01.065', 'VMTS13.01.078', 'VMTS13.01.106', 'VMTS13.01.086']

ex = vw.example(
    {"a": feats + ["none"],
     "b": map(lambda x: x+"_in_cache", feats),
     "c": ["all clear", "stay"],
    },
    labelType=vw.lCostSensitive)
for i, f in enumerate(feats):
    fid = ex.feature("a", i)
    print f, fid, vw.get_weight(fid)
    cachef = f + "_in_cache"
    cachefid = ex.feature("b", i)
    print cachef, cachefid, vw.get_weight(cachefid)

fid = ex.feature("a", len(feats))
print "none", fid, vw.get_weight(fid)

#print ex.sum_feat_sq("a")
#print ex.sum_feat_sq("b")
#print ex.sum_feat_sq("c")
print "all_clear", vw.get_weight(ex.feature("c", 0))
print "stay", vw.get_weight(ex.feature("c", 1))
#    print f, ex.feature_weight("b", i)
#print "all clear", ex.feature_weight("c", 0)
#print "stay", ex.feature_weight("c", 1)

my_iter = new_df_iter()
doc = next(my_iter)

tot_docs = 0
for i in sequence:
    if i < len(doc):
        print doc[i]["sent text"]
        print doc[i]["nuggets"]
        del doc[i]
    else:
        tot_docs += 1
        doc = next(my_iter)
        print "next", doc[0]["update id"]
print tot_docs 
vw.finish()
