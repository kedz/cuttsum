import pyvw
import numpy as np
import pandas as pd

class _SearchBase(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.IS_LDF ) # | sch.AUTO_CONDITION_FEATURES )
        self._with_scores = False

    def list_namespaces(self):
        return ["a", "b", "c", "d", "e", "f", "g", "n"]

    def get_example_score(self, ex):
        w_sum = 0
        for ns in self.list_namespaces():
            w_sq = 0
            for i in xrange(ex.num_features_in(ns)):
                w = self.vw.get_weight(ex.feature(ns, i)) * ex.feature_weight(ns, i)
                w_sum += w
        return w_sum
    def get_namespace_scores(self, ex, ns_map=None):
        scores = {}
        for ns in self.list_namespaces():
            score = 0
            for i in xrange(ex.num_features_in(ns)):
                score += self.vw.get_weight(ex.feature(ns, i)) * ex.feature_weight(ns, i)
            if ns_map is None:
                scores[ns] = score    
            else:
                scores[ns_map(ns)] = score
        return scores


    def make_select_example(self, sent, sents, df, cache):
        pass

    def make_next_example(self, sents, df, cache, is_oracle): 
        pass

    def setup_cache(self):
        pass

    def update_cache(self, pred, sents, df, cache):
        pass
    
    def get_feature_weights(self, dataframes):
        pass

    def predict_with_scores(self, instance):
        self._with_scores = True
        seq, df = self.predict(instance)
        self._with_scores = False
        return seq, df
    
    def _run(self, (event, docs)):
        nuggets = set()
        cache = self.setup_cache()
        output = []
        n = 0
        loss = 0

        if self._with_scores is True:
            score_data = []

        for doc in docs:

            sents = range(len(doc))

            while 1:

                n += 1
                # Create some examples, one for each sentence.
                examples = [self.make_select_example(sent, sents, doc, cache)
                            for sent in sents]
                    #examples = [ex() for ex in examples]

                # Create a final example for the option "next document".
                # This example has the feature "all_clear" if the max gain
                # of adding any current sentence to the summary is 0.
                # Otherwise the feature "stay" is active.
                gain = doc.iloc[sents]["nuggets"].apply(
                    lambda x: len(x.difference(nuggets))).values

               
                # Compute oracle. If max gain > 0, oracle always picks the 
                # sentence with the max gain. Else, oracle picks the last 
                # example which is the "next document" option.
                if len(sents) > 0:
                    oracle = np.argmax(gain) 
                    oracle_gain = gain[oracle] 
                    if oracle_gain == 0:
                        oracle = len(sents)
                else:
                    oracle = 0
                    oracle_gain = 0
            
                oracle_is_next = oracle == len(sents)
                next_ex = self.make_next_example(
                    sents, doc, cache, oracle_is_next)
                examples.append(next_ex)

                # Make prediction.
                pred = self.sch.predict(
                    examples=examples,
                    my_tag=n,
                    oracle=oracle,
                    condition=[], # (n-1, "p"), ])
                )
                output.append(pred)
                
                if oracle_is_next:
                    if pred != oracle:
                        loss += 1
                else:
                    if pred < len(sents):
                        loss += (oracle_gain - gain[pred])    
                    else:
                        missed = set([nug for ns in doc.iloc[sents]["nuggets"].tolist()
                                      for nug in ns if nug not in nuggets])
                        loss += len(missed)

                if self._with_scores is True:
                    scores = np.array([self.get_example_score(ex) 
                                       for ex in examples])
                    ns_scores = self.get_namespace_scores(examples[pred], ns_map=lambda x: "l_" + x)
                    o_ns_scores = self.get_namespace_scores(examples[oracle], ns_map=lambda x: "o_" + x)
                    if scores.shape[0] > 1:
                        select_scores = scores[0:-1]
                        max_select = np.max(select_scores)
                        min_select = np.min(select_scores)
                        avg_select = np.mean(select_scores)
                        med_select = np.median(select_scores)
                    else:
                        max_select = 0
                        min_select = 0
                        avg_select = 0
                        med_select = 0
                    score_data.append({
                        "max select score": max_select,
                        "min select score": min_select,
                        "avg select score": avg_select,
                        "median select score": med_select,
                        "next score": scores[-1],                
                    })
                    score_data[-1].update(ns_scores)
                    score_data[-1].update(o_ns_scores)
                    assert np.min(scores) == scores[pred]
                if pred < len(sents):
                    cache = self.update_cache(pred, sents, doc, cache)
                    nuggets.update(doc.iloc[pred]["nuggets"])
                    del sents[pred]
                                        
                elif pred == len(sents):
                    break
        
        self.sch.loss(loss)
        if self._with_scores:
            return output, pd.DataFrame(score_data, 
                columns=["min select score", "max select score", 
                         "avg select score", "median select score",
                         "next score"] + map(lambda x: "l_" + x, self.list_namespaces()) + map(lambda x: "o_" + x, self.list_namespaces()))
        else:
            return output

