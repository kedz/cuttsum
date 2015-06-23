import pyvw
import numpy as np

class _SearchBase(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.AUTO_HAMMING_LOSS | sch.IS_LDF ) # | sch.AUTO_CONDITION_FEATURES )

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

    def _run(self, (event, docs)):
        nuggets = set()
        cache = self.setup_cache()
        output = []
        n = 0

        for doc in docs:

            sents = range(len(doc))

            while 1:

                n += 1
                # Create some examples, one for each sentence.
                examples = [self.make_select_example(sent, sents, doc, cache)
                            for sent in sents]
                

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
            
                next_ex = self.make_next_example(
                    sents, doc, cache, oracle == len(sents))
                examples.append(next_ex)

                # Make prediction.
                pred = self.sch.predict(
                    examples=examples,
                    my_tag=n,
                    oracle=oracle,
                    condition=[], # (n-1, "p"), ])
                )
                output.append(pred)

                if pred < len(sents):
                    cache = self.update_cache(pred, sents, doc, cache)
                    nuggets.update(doc.iloc[pred]["nuggets"])
                    del sents[pred]
                                        
                elif pred == len(sents):
                    break
   
        return output

