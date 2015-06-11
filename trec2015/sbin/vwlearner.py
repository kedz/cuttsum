import cuttsum.events
import cuttsum.corpora
from cuttsum.pipeline import ArticlesResource
import pyvw
import numpy as np



class UpdateSummarizer(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.AUTO_HAMMING_LOSS | sch.IS_LDF | sch.AUTO_CONDITION_FEATURES )
        self.ncache = set()
        self.updates = []

    def make_example(self, sentence, ncache):
        nuggets = sentence[1]
        ex = self.vw.example(
            {"a": [nid for nid in nuggets],
             "b": [nid + "_in_cache" for nid in self.ncache]},
            labelType=self.vw.lCostSensitive)
        return ex

    def _run(self, doc_orig):
        doc = list(doc_orig)
        output = []
        ncache = set()
        n = 0
        while len(doc) > 0:
            n += 1
            print ncache
            examples = [self.make_example(sent, ncache) for sent in doc]
            gain = map(lambda x: len(x[1].difference(ncache)), doc)
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
                ncache.update(doc[pred][1])
                print "Adding sent", doc[pred][0]
                del doc[pred]
                                    
            if pred == len(doc):
                break
            if oracle == len(doc):
                break
   
        return output

    def _run2(self, df):
        df = df.copy(True)

        output = []
         
        n = 0
        while 1: 
            print self.ncache
            n += 1       
            examples = df["nuggets"].apply(self.make_example).tolist()
            print "THRE ARE", len(examples), "examples"
            #for i, ex in enumerate(examples, 1):
            #    ex.set_label_string(str(i) + ":0")

            gain = df["nuggets"].apply(lambda x: len(x.difference(self.ncache))).values
            oracle = gain.argmax()
            oracle_gain = gain[oracle]
            if oracle_gain == 0:
                oracle = len(examples)
            examples.append(
                self.vw.example(
                    {"c": ["all_clear" if gain.sum() == 0 else "stay"],},
                    labelType=self.vw.lCostSensitive))
    
            pred = self.sch.predict(
                examples=examples,
                my_tag=n,
                oracle=oracle,
                condition=[ (n-1, "p"), ])
            output.append(pred)

            print "PRED", pred
            if pred < len(df):
                print df.loc[pred, "sent text"]
            else:
                print "EXIT"
            print "ORACLE", oracle
            if oracle < len(df):
                print df.loc[oracle, "sent text"]
            else:
                print "EXIT"

#            for ex in examples: ex.finish()

            if pred < len(df):
                self.ncache.update(df.loc[pred, "nuggets"])
                self.updates.append(df.loc[pred])
                print "Adding", self.updates[-1]["sent text"]
                I = range(pred) + range(pred + 1, len(df))
                assert pred not in I
                
                print I
                
                df2 = df.ix[I]
                assert len(df) - 1 == len(df2)
                df = df2 
                df.reset_index(inplace=True, drop=True)

            if pred == len(examples) - 1:
                break
            if oracle == len(examples) - 1:
                break
            if len(df) == 0:
                break
        print "I am outputing"
        return output

event = cuttsum.events.get_events()[0]
corpus = cuttsum.corpora.EnglishAndUnknown2013()
extractor = "goose" 
res = ArticlesResource()
k = 5
m = 10
cache = {}

dfiter = res.dataframe_iter(
    event, corpus, extractor, include_matches="soft")

def dfwrap(dfiter):
    for df in dfiter:
        items = []
        for _, row in df.iterrows():
            items.append(
                (row["sent text"], row["nuggets"]))
        yield items


doc = next(dfwrap(dfiter))

vw = pyvw.vw("--search 0 --csoaa_ldf m --search_task hook --ring_size 1024 --quiet")
task = vw.init_search_task(UpdateSummarizer)
task.learn([doc])
sequence = task.predict(doc)
for i in sequence:
    if i < len(doc):
        print doc[i][0]
        print doc[i][1]
        del doc[i]
    else:
        print "next"

 

