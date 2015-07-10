import pyvw
from cuttsum.l2s._base import _SearchBase
import pandas as pd

class SelectBasicNextBias(_SearchBase):
    
    def setup_cache(self):
        return None

    def basic_cols(self):
        return [
    "BASIC length", "BASIC char length", "BASIC doc position", 
    "BASIC all caps ratio", "BASIC upper ratio", "BASIC lower ratio",
    "BASIC punc ratio", "BASIC person ratio", "BASIC organization ratio",
    "BASIC date ratio", "BASIC time ratio", "BASIC duration ratio",
    "BASIC number ratio", "BASIC ordinal ratio", "BASIC percent ratio",
    "BASIC money ratio", "BASIC set ratio", "BASIC misc ratio"]



    def update_cache(self, pred, sents, df, cache):
        return cache



    def make_select_example(self, sent, sents, df, cache):
        return self.example(lambda: {
            "b": [x for x in df.iloc[sent][self.basic_cols()].iteritems()],},
            labelType=self.vw.lCostSensitive)

    def make_next_example(self, sents, df, cache, is_oracle): 
        return self.example(lambda: {"n": ["bias"],},
                labelType=self.vw.lCostSensitive)

    def get_feature_weights(self, dataframes):
        ex = self.vw.example(
            {"b": self.basic_cols(),
             "n": ["bias"],
            },
            labelType=self.vw.lCostSensitive)
        fw = []
        for i, feat in enumerate(self.basic_cols()):
            w = self.vw.get_weight(ex.feature("b", i))
            fw.append(("b:" + feat, w))
        fw.append(("n:bias", self.vw.get_weight(ex.feature("n", 0))))
        fw.sort(key=lambda x: x[1])
        return fw             

class SelectBasicNextBiasDocAvg(_SearchBase):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.IS_LDF )
        self._with_scores = False


    def setup_cache(self):
        return pd.DataFrame(columns=self.basic_cols())

    def basic_cols(self):
        return [
    "BASIC length", "BASIC char length", "BASIC doc position", 
    "BASIC all caps ratio", "BASIC upper ratio", "BASIC lower ratio",
    "BASIC punc ratio", "BASIC person ratio", "BASIC organization ratio",
    "BASIC date ratio", "BASIC time ratio", "BASIC duration ratio",
    "BASIC number ratio", "BASIC ordinal ratio", "BASIC percent ratio",
    "BASIC money ratio", "BASIC set ratio", "BASIC misc ratio",
    "LM domain avg lp", "LM gw avg lp"]

    def update_cache(self, pred, sents, df, cache):
        series = df.iloc[pred][self.basic_cols()]
        cache = cache.append(series, ignore_index=True)
        return cache

    def make_select_example(self, sent, sents, df, cache):
        if len(cache) > 0:
            return self.example(lambda: {
                "a": [x for x in df.iloc[sent][self.basic_cols()].iteritems()],
                "b": [x for x in df.iloc[sents][self.basic_cols()].mean().iteritems()],
                "c": [x for x in cache.mean().iteritems()]
                },
                labelType=self.vw.lCostSensitive)
        else:
            return self.example(lambda: {
                "a": [x for x in df.iloc[sent][self.basic_cols()].iteritems()],
                "b": [x for x in df.iloc[sents][self.basic_cols()].mean().iteritems()],
                },
                labelType=self.vw.lCostSensitive)

    def make_next_example(self, sents, df, cache, is_oracle): 
        if len(sents) > 0 and len(cache) > 0:
            return self.example(lambda: {
                "d": ["bias"],
                "e": [x for x in df.iloc[sents][
                    self.basic_cols()].mean().iteritems()],
                "f": [x for x in cache.mean().iteritems()]
                },
                labelType=self.vw.lCostSensitive)
        elif len(sents) > 0 and len(cache) == 0:
            return self.example(lambda: {
                "d": ["bias"],
                "e": [x for x in df.iloc[sents][
                    self.basic_cols()].mean().iteritems()],
                },
                labelType=self.vw.lCostSensitive)
        elif len(sents) == 0 and len(cache) > 0:
            return self.example(lambda: {
                "d": ["bias"],
                "f": [x for x in cache.mean().iteritems()]
                },
                labelType=self.vw.lCostSensitive)
        else:
            return self.example(lambda: {
                "d": ["bias"],
                },
                labelType=self.vw.lCostSensitive)

    def get_feature_weights(self, dataframes):
        ex = self.vw.example(
            {"a": self.basic_cols(),
             "b": self.basic_cols(),
             "c": self.basic_cols(),
             "d": ["bias"],
             "e": self.basic_cols(),
             "f": self.basic_cols(),
            },
            labelType=self.vw.lCostSensitive)
        fw = []
        for i, feat in enumerate(self.basic_cols()):
            w = self.vw.get_weight(ex.feature("a", i))
            fw.append(("a:" + feat, w))
        for i, feat in enumerate(self.basic_cols()):
            w = self.vw.get_weight(ex.feature("b", i))
            fw.append(("b:" + feat, w))
        for i, feat in enumerate(self.basic_cols()):
            w = self.vw.get_weight(ex.feature("c", i))
            fw.append(("c:" + feat, w))
        for i, feat in enumerate(self.basic_cols()):
            w = self.vw.get_weight(ex.feature("e", i))
            fw.append(("e:" + feat, w))
        for i, feat in enumerate(self.basic_cols()):
            w = self.vw.get_weight(ex.feature("f", i))
            fw.append(("f:" + feat, w))

        fw.append(("d:bias", self.vw.get_weight(ex.feature("d", 0))))
        fw.sort(key=lambda x: x[1])
        return fw          
