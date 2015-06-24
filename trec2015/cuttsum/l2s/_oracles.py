from cuttsum.l2s._base import _SearchBase

class SelectLexNextOracle(_SearchBase):
    
    def setup_cache(self):
        return set()

    def update_cache(self, pred, sents, df, cache):
        cache.update(df.iloc[pred]["lemmas stopped"])
        return cache

    def make_select_example(self, sent, sents, df, cache):
        tokens = df.iloc[sent]["lemmas stopped"]
        cache_feats = [tok for tok in tokens if tok in cache]
        ex = self.example(lambda:
            {"a": tokens if len(tokens) > 0 else ["__none__"],
             "b": cache_feats if len(cache_feats) > 0 else ["__none__"],
            },
            labelType=self.vw.lCostSensitive)
        return ex

    def make_next_example(self, sents, df, cache, is_oracle): 
        if is_oracle:
            return self.example(lambda: {"c": ["all_clear"],},
                labelType=self.vw.lCostSensitive)
        else:
            return self.example(lambda: {"c": ["stay"],},
                labelType=self.vw.lCostSensitive)

    def get_feature_weights(self, dataframes):
        vocab = set([w for df in dataframes
                     for words in df["lemmas stopped"].tolist()
                     for w in words])
        vocab = list(vocab)
        lexical_feats = vocab + ["__none__"]
        lexical_cache_feats = vocab
        for w in vocab:
            assert not isinstance(w, unicode)
        ex = self.vw.example(
            {"a": lexical_feats,
             "b": lexical_cache_feats,
             "c": ["stay", "all_clear"],
            },
            labelType=self.vw.lCostSensitive)
        fw = []
        for i, feat in enumerate(lexical_feats):
            w = self.vw.get_weight(ex.feature("a", i))
            fw.append(("a:" + feat, w))
        for i, feat in enumerate(lexical_cache_feats):
            w = self.vw.get_weight(ex.feature("b", i))
            fw.append(("b:" + feat, w))
        fw.append(("c:stay", self.vw.get_weight(ex.feature("c", 0))))
        fw.append(("c:all_clear", self.vw.get_weight(ex.feature("c", 0))))
        fw.sort(key=lambda x: x[1])
        return fw             


class SelectLexNextLex(SelectLexNextOracle):

    def make_next_example(self, sents, df, cache, is_oracle): 
        if len(sents) > 0:
            tokens = set([w for words in df.iloc[sents]["lemmas stopped"].tolist()
                          for w in words])
            tokens = list(tokens) 
            return self.example(lambda: {"c": tokens,},
                labelType=self.vw.lCostSensitive)
        else:
            return self.example(lambda: {"c": ["__none__"],},
                labelType=self.vw.lCostSensitive)

    def get_feature_weights(self, dataframes):
        vocab = set([w for df in dataframes
                     for words in df["lemmas stopped"].tolist()
                     for w in words])
        vocab = list(vocab)
        lexical_feats = vocab + ["__none__"]
        lexical_cache_feats = vocab
        for w in vocab:
            assert not isinstance(w, unicode)
        ex = self.vw.example(
            {"a": lexical_feats,
             "b": lexical_cache_feats,
             "c": lexical_feats,
            },
            labelType=self.vw.lCostSensitive)
        fw = []
        for i, feat in enumerate(lexical_feats):
            w = self.vw.get_weight(ex.feature("a", i))
            fw.append(("a:" + feat, w))
        for i, feat in enumerate(lexical_cache_feats):
            w = self.vw.get_weight(ex.feature("b", i))
            fw.append(("b:" + feat, w))
        for i, feat in enumerate(lexical_feats):
            w = self.vw.get_weight(ex.feature("c", i))
            fw.append(("c:" + feat, w))

        fw.sort(key=lambda x: x[1])
        return fw             

class SelectLexNextLexCache(SelectLexNextOracle):

    def make_next_example(self, sents, df, cache, is_oracle): 
        if len(sents) > 0:
            tokens = set([w for words in df.iloc[sents]["lemmas stopped"].tolist()
                          for w in words])

            tokens = list(tokens) 
            cache_feats = [tok for tok in tokens if tok in cache]
            return self.example(lambda: {
                "c": tokens if len(tokens) > 0 else "__none__", 
                "d": cache_feats if len(cache_feats) > 0 else "__none__"},
                labelType=self.vw.lCostSensitive)
        else:
            return self.example(lambda: {"c": ["__none__"], "d": ["__none__"]},
                labelType=self.vw.lCostSensitive)

    def get_feature_weights(self, dataframes):
        vocab = set([w for df in dataframes
                     for words in df["lemmas stopped"].tolist()
                     for w in words])
        vocab = list(vocab)
        lexical_feats = vocab + ["__none__"]
        lexical_cache_feats = vocab
        for w in vocab:
            assert not isinstance(w, unicode)
        ex = self.vw.example(
            {"a": lexical_feats,
             "b": lexical_cache_feats,
             "c": lexical_feats,
             "d": lexical_feats,
            },
            labelType=self.vw.lCostSensitive)
        fw = []
        for i, feat in enumerate(lexical_feats):
            w = self.vw.get_weight(ex.feature("a", i))
            fw.append(("a:" + feat, w))
        for i, feat in enumerate(lexical_cache_feats):
            w = self.vw.get_weight(ex.feature("b", i))
            fw.append(("b:" + feat, w))
        for i, feat in enumerate(lexical_feats):
            w = self.vw.get_weight(ex.feature("c", i))
            fw.append(("c:" + feat, w))
        for i, feat in enumerate(lexical_feats):
            w = self.vw.get_weight(ex.feature("d", i))
            fw.append(("d:" + feat, w))

        fw.sort(key=lambda x: x[1])
        return fw             


class SelectLexGenericNextOracle(SelectLexNextOracle):
    
    def basic_cols(self):
        return [
            "BASIC length", "BASIC char length", "BASIC doc position", 
            "BASIC all caps ratio", "BASIC upper ratio", "BASIC lower ratio",
            "BASIC punc ratio", "BASIC person ratio", "BASIC organization ratio",
            "BASIC date ratio", "BASIC time ratio", "BASIC duration ratio",
            "BASIC number ratio", "BASIC ordinal ratio", "BASIC percent ratio",
            "BASIC money ratio", "BASIC set ratio", "BASIC misc ratio"]
    
    def lm_cols(self):
        return ["LM domain avg lp", 
            "LM gw avg lp"]

    def setup_cache(self):
        return set()

    def update_cache(self, pred, sents, df, cache):
        cache.update(df.iloc[pred]["lemmas stopped"])
        return cache

    def make_select_example(self, sent, sents, df, cache):
        tokens = df.iloc[sent]["lemmas stopped"]
        cache_feats = [tok for tok in tokens if tok in cache]
        gen_cols = self.basic_cols() + self.lm_cols() 
        ex = self.example(lambda:
            {"a": tokens if len(tokens) > 0 else ["__none__"],
             "b": cache_feats if len(cache_feats) > 0 else ["__none__"],
             "g": [x for x in df.iloc[sent][gen_cols].iteritems()],
            },
            labelType=self.vw.lCostSensitive)
        return ex

    def make_next_example(self, sents, df, cache, is_oracle): 
        if is_oracle:
            return self.example(lambda: {"c": ["all_clear"],},
                labelType=self.vw.lCostSensitive)
        else:
            return self.example(lambda: {"c": ["stay"],},
                labelType=self.vw.lCostSensitive)

    def get_feature_weights(self, dataframes):
        vocab = set([w for df in dataframes
                     for words in df["lemmas stopped"].tolist()
                     for w in words])
        vocab = list(vocab)
        lexical_feats = vocab + ["__none__"]
        lexical_cache_feats = vocab
        generics = self.basic_cols() + self.lm_cols()
        for w in vocab:
            assert not isinstance(w, unicode)
        ex = self.vw.example(
            {"a": lexical_feats,
             "b": lexical_cache_feats,
             "g": [(feat, 1) for feat in generics],
             "c": ["stay", "all_clear"],
            },
            labelType=self.vw.lCostSensitive)
        fw = []
        for i, feat in enumerate(lexical_feats):
            w = self.vw.get_weight(ex.feature("a", i))
            fw.append(("a:" + feat, w))
        for i, feat in enumerate(lexical_cache_feats):
            w = self.vw.get_weight(ex.feature("b", i))
            fw.append(("b:" + feat, w))
        for i, feat in enumerate(generics):
            w = self.vw.get_weight(ex.feature("g", i))
            fw.append(("g:" + feat, w))


        fw.append(("c:stay", self.vw.get_weight(ex.feature("c", 0))))
        fw.append(("c:all_clear", self.vw.get_weight(ex.feature("c", 0))))
        fw.sort(key=lambda x: x[1])
        return fw             


