import cuttsum.events
import cuttsum.corpora
from cuttsum.pipeline import InputStreamResource
from collections import defaultdict
import pyvw
#from mpi4py import MPI
from cuttsum.misc import enum
import sys
import random
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_rows', 50000)
from itertools import izip
from datetime import datetime

tags = enum("READY", "WORKER_START", "WORKER_STOP", "SENDING_RESULT")

class DocumentClassifier(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.IS_LDF )
        self._with_scores = False

    def score(self, instance):
        self._with_scores = True
        seq, df = self.predict(instance)
        good_cols = []
        for col in df.columns:           
            if col.startswith("NEXT_") or col.startswith("SELECT_"):
                if not (df[col] == 0).all():
                    good_cols.append(col)
            else:
                good_cols.append(col)
        self._with_scores = False
        return df[good_cols]

    def list_namespaces(self):
        return [fgroup.ns() for fgroup in self._feature_groups]

    def ns_map(self, ns):
        pass

    def register_features(self, feature_groups):
        self._feature_groups = feature_groups

    def setup_NEXT_cache(self, stream_df):
        cache = {}
        cache["updates"] = None
        for fgroup in self._feature_groups:
            fgroup.setup_cache(cache)
        #for fgroup in self._feature_groups:
        #    key = fgroup.ns()
        #    fdict = {}
        #    for f in fgroup.features():
        #        fdict[f] = np.zeros((len(stream_df),))
        #    cache[key] = fdict
        return cache
   
    def update_cache(self, sent, sents, df, cache):
        if cache["updates"] is None:
            cache["updates"] = df.iloc[sent].to_frame().transpose()
        else:
            cache["updates"] = pd.concat(
                [cache["updates"], df.iloc[sent].to_frame().transpose()])

        return cache


    def make_SELECT_examples(self, sents, df, cache, output, df_stream, 
        stream_idx, meta):
        
        feature_maps = [dict() for i in sents]
        for fgroup in self._feature_groups:
            fgroup.make_SELECT_features(feature_maps, sents, df, cache, output, df_stream,
                stream_idx, meta)
        exs = [self.example(feature_map, labelType=self.vw.lCostSensitive)
               for feature_map in feature_maps]
        if self._with_scores:
            for i, fmap in enumerate(feature_maps):
                scores = self.get_scores(exs[i])
                print df.iloc[sents[i]]["pretty text"]
                print fmap
                print scores[0], scores[1]
        return exs

    def make_NEXT_examples(self, select_examples, hours_since_start):

        n_examples = len(select_examples)
        score_dicts = [self.get_scores(ex)[1] for ex in select_examples]
        keys = score_dicts[0].keys()
        keys.sort()

        next_feats = []
        for key in keys:
            scores = [sd[key] for sd in score_dicts]
            scores = np.array(scores)

            if scores.min() < 0:
                scores += np.fabs(scores.min())
            Z = np.sum(scores)
            if Z == 0:
                scores.fill(1)
                Z = scores.shape[0]
            ex_dist = scores / Z

            uni_dist = np.ones(n_examples) / n_examples

            if self._with_scores is True:
                print scores
                print "exa", ex_dist
                print "uni", uni_dist
            kl = np.sum(
                [ex_dist[i] * (np.log(ex_dist[i]) - np.log(uni_dist[i])) 
                 for i in xrange(n_examples)])
             
            bc = np.sum(
                [np.sqrt(ex_dist[i] * uni_dist[i]) for i in xrange(n_examples)])
            next_feats.append((key, bc * 1. / hours_since_start))
        fmap = {
            #"z": [("kl", kl), ("bc", bc)]
            "z": next_feats #+ [("hours", 1./ hours_since_start)]
        }

        x = self.example(fmap, labelType=self.vw.lCostSensitive)
        if self._with_scores is True:
            print fmap
            print self.get_scores(x)
        return x

    def get_scores(self, ex, prefix=""):
        scores = {}

        tot_score = 0
        for ns in [fgroup.ns() for fgroup in self._feature_groups] + ["z"]:
            fscore = 0
            for i in xrange(ex.num_features_in(ns)):
                fscore += self.vw.get_weight(ex.feature(ns, i)) \
                    * ex.feature_weight(ns, i)
            scores[prefix + ns] = fscore
            tot_score += fscore
        return tot_score, scores    

    def get_feature_values_weights(self, ex, prefix=""):
        scores = {}
        for fgroup in self._feature_groups:
            for i in xrange(ex.num_features_in(fgroup.ns())):
                fname = fgroup.features()[i]
                w = self.vw.get_weight(ex.feature(fgroup.ns(), i))
                v = ex.feature_weight(fgroup.ns(), i)
                key = prefix + fgroup.ns() + "_" + fname
                scores[key + "_w"] = w
                scores[key + "_v"] = v
        return scores

    def get_namespaces(self, prefix=""):
        return [prefix + fgroup.ns() 
                for fgroup in self._feature_groups]

    def compute_gain(self, sents, df, nuggets):
        gain = df.iloc[sents]["nuggets"].apply(
            lambda x: len(x.difference(nuggets))).values
        return gain

    def get_oracle(self, gain):
        greedy_best_select = np.argmax(gain)
        max_gain = gain[greedy_best_select]
        if max_gain == 0:
            return gain.shape[0]
        else:
            return greedy_best_select ### NEXT=0 SELECT=oracle

    def _run(self, (event, df_stream)):
        nuggets = set()
        cache = self.setup_NEXT_cache(df_stream)
        output = []
        loss_val = 19.8
        n = 0
        loss = 0
        tot_correct = 0

        tot_next = 0
        tot_select = 0
        tot_correct_next = 0
        tot_correct_select = 0

        meta = {"sents_selected_in": {}}
        if self._with_scores is True:
            score_data = []

        event.start
        for doc_idx, doc in enumerate(df_stream):
            sents = range(len(doc))
            dur = datetime.utcfromtimestamp(doc["timestamp"][0]) - event.start
            hours_since_start = dur.total_seconds() / (3600.)
            if self._with_scores:
                print dur, hours_since_start
            meta["sents_selected_in"][doc_idx] = []
            while 1:
                n += 1

                if self.sch.predict_needs_example():
                    select_examples = self.make_SELECT_examples(
                        sents, doc, cache, output, df_stream, doc_idx, meta)
                    next_example = self.make_NEXT_examples(select_examples, hours_since_start)
                    examples = select_examples + [next_example]
                else:
                    examples = [None] * (len(sents) + 1)

                gain = self.compute_gain(sents, doc, nuggets)
                oracle = self.get_oracle(gain)


                # Make prediction.
                pred = self.sch.predict(
                    examples=examples,
                    my_tag=n,
                    oracle=oracle,
                    condition=[], # (n-1, "p"), ])
                )
                output.append(pred)
            
                #if self._with_scores:
                #    print "\n".join(doc.iloc[sents]["pretty text"].tolist())
                #    print sents
                #    print pred
                #    print

                pred_gain = gain[pred] if pred < len(sents) else 0
                
                if oracle != pred and pred_gain == 0:
                    
                    if pred == len(sents):
                        loss += loss_val
                    else:
                        loss += 1
                else:
                    tot_correct += 1   
                    if pred == len(sents):
                        tot_correct_next += 1
                    else:             
                        tot_correct_select += 1
                if oracle == len(sents):
                    tot_next += 1
                else:
                    tot_select += 1

                if self._with_scores:
                    #select_score, select_ns_scores = self.get_scores(
                    #    examples[0], "SELECT_")
                    #print select_ns_scores 
                    #next_score, next_ns_scores = self.get_scores(
                    #    examples[1], "NEXT_")
                    #print next_ns_scores
                    if pred < len(sents):
                        update_text = doc.iloc[sents[pred]]["pretty text"]
                    else:
                        update_text = None

                    if oracle < len(sents):
                        oracle_text = doc.iloc[sents[oracle]]["pretty text"]
                    else:
                        oracle_text = None


                    d = {
                        "action": "NEXT" if pred == len(sents) else "SELECT_{}".format(pred),
                        "oracle": "NEXT" if oracle == len(sents) else "SELECT_{}".format(oracle),
                        "acc": float(tot_correct) / n,
                        "doc id": doc_idx,
                        "oracle text": oracle_text,
                        "pred text": update_text,
                      #  "SELECT": select_score,
                        #"NEXT": next_score,
                        #"update": doc.iloc[sents[oracle_SELECT]]["pretty text"] if oracle_SELECT is not None else None,
                        "correct next": tot_correct_next,
                        "tot next": tot_next,
                        "correct select": tot_correct_select,
                        "tot select": tot_select,
                        "wgt acc": (loss_val * tot_correct_select + tot_correct_next) / float(loss_val * tot_select + tot_next),
                        "nuggets": len(nuggets)
                        
                    }
                    #d.update(select_ns_scores)
                    #d.update(next_ns_scores)
                    #d.update(self.get_feature_values_weights(examples[1]))
                    score_data.append(d) 
                    #print d["oracle"], d["action"], gain
                    #if cache is not None:
                    #    print cache[["pretty text", "nuggets",]]
                    #print doc.iloc[sents][["pretty text", "nuggets",]]
                    #inp_nuggets = list(set([ng for ns in doc.iloc[sents]["nuggets"].tolist()
                    #                        for ng in ns]))
                    #print "input nuggets"
                    #inp_nuggets.sort()
                    #for ng in inp_nuggets:
                    #    print ng,
                    #print
                    #if cache is not None:
                    #    print "cache nuggets"
                    #    upd_nuggets = list(set([ng for ns in cache["nuggets"].tolist()
                    #                            for ng in ns]))
                    #else:
                    #    upd_nuggets = []
                    #upd_nuggets.sort()
                    #for ng in upd_nuggets:
                    #    print ng,
                    #print
                    #print
                #if self._with_scores is True:
                
                if pred < len(sents):
                    #if oracle_SELECT is None:
                    #    I = range(len(sents))
                    #    random.shuffle(I)
                    #    select = I[0]
                    #else:
                    #    select = oracle_SELECT
                    #if self._with_scores:
                    #    print select, oracle_SELECT, sents[select], sents
                    #    print doc.iloc[sents[select]][["pretty text", "nuggets"]]

                    cache = self.update_cache(sents[pred], sents, doc, cache)
                    nuggets.update(doc.iloc[sents[pred]]["nuggets"])
                    #meta["sents_selected_in"][doc_idx].append(sents[select])
                    del sents[pred]

                if pred == len(sents) or len(sents) == 0:
                    tot = 0 
                    if "wc" in cache:
                        for words in doc["lemmas stopped"].tolist():
                            for word in words:
                                cache["wc"][word] += 1
                            tot += len(words)
                        cache["wc"]["__total__"] += float(tot)

                    break                      
                
                 
                

                #elif pred == len(sents):
                #    break
        
        self.sch.loss(loss)
        if self._with_scores:
            
            columns=["correct next", "tot next", "correct select", 
                     "tot select", "action", "oracle", "wgt acc", "acc",
                     "doc id", "nuggets", "pred text", "oracle text"] 
            
            #for fgroup in self._feature_groups:
            #    columns.append("NEXT_" + fgroup.ns())
            #    for fname in fgroup.features():
            #        columns.append(fgroup.ns() + "_" + fname + "_v")
            #        columns.append(fgroup.ns() + "_" + fname + "_w")

            return output, pd.DataFrame(score_data, columns=columns)
                
#columns=["correct next", "tot next", "correct select", "tot select", "action", "oracle", "wgt acc", "acc", "doc id",] \
           #         + ["NEXT"] + self.get_namespaces(prefix="NEXT_") + ["update"])
#                columns=["min select score", "max select score", 
#                         "avg select score", "median select score",
#                         "next score"] + map(lambda x: "l_" + x, self.list_namespaces()) + map(lambda x: "o_" + x, self.list_namespaces()))
        else:        
            return output


class FeatureGroup(object):

    def ns(self):
        pass

    def features(self):
        pass

    def initialize_vw(self, search_task):
        print self.name()
        fdict = {self.ns(): [w for w in self.features()]}
        ex = search_task.example(
            fdict)#, labelType=search_task.vw.lCostSensitive) 
        ex.push_feature_dict(search_task.vw, fdict)
        self._ex = ex

    def get_weights(self, search_task):
        
        feature_weights = {}
        for i, f in enumerate(self.features()):
            w = search_task.vw.get_weight(self._ex.feature(self.ns(), i))
            feature_weights[f] = w
        return feature_weights

    def set_weights(self, search_task, feature_weights):
        for i, f in enumerate(self.features()):
            w = feature_weights[f]
            search_task.vw.set_weight(self._ex.feature(self.ns(), i), 0, w)

class OracleFeatures(FeatureGroup):

    def name(self):
        return "oracle"

    def ns(self):
        return "o"

    def features(self):
        return ["pos_gain", "0_gain"]

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):

        inp_nuggets = set([n for ns in df.iloc[sents]["nuggets"].tolist()
                           for n in ns])

        if cache is not None:
            upd_nuggets = set([n for ns in cache["nuggets"].tolist()
                               for n in ns])
        else:
            upd_nuggets = set()

        if len(inp_nuggets - upd_nuggets) > 0:
            next_f[self.ns()] = ["pos_gain"]
        else:
            next_f[self.ns()] = ["0_gain"]

class MaxSim(FeatureGroup):
    def name(self):
        return "max_sim"

    def ns(self):
        return "s"

    def features(self):
        return ["max_sim"]

    def setup_cache(self, cache):
        cache[self.name()] = []
        cache["X"] = None

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):

        hasher = FeatureHasher()
        counts = {}
        for words in df.iloc[sents]["lemmas stopped"].tolist():
            for word in words:
                counts[word] = counts.get(word, 0) + 1
        X_d = hasher.transform([counts])

        if cache["X"] is None:
            cache["X"] = X_d
            cache[self.name()].append(0)
            return #(self.name(), 0)
        if stream_idx == 0:
            return #(self.name(), 0)
            
        X_h = cache["X"][:stream_idx]

        K = cosine_similarity(X_h, X_d)
        fval = K.max()

        if len(sents) == len(df):
            cache[self.name()].append(fval)
            cache["X"] = scipy.sparse.vstack([cache["X"], X_d])

        next_f[self.ns()] = [(self.features()[0], fval)]

class MaxInputUpdateSim(FeatureGroup):
    def name(self):
        return "max_iu_sim"

    def ns(self):
        return "m"

    def features(self):
        return ["max_iu_sim"]

    def setup_cache(self, cache):
        pass # cache[self.name()] = []


    def make_SELECT_features(self, feature_maps, sents, df, cache, output,
            df_stream, stream_idx, meta):

        if cache["updates"] is None:
            #next_f[self.ns()] = ["empty_cache"]
            return

        hasher = FeatureHasher()
        i_counts = []
        for words in df.iloc[sents]["lemmas stopped"].tolist():
            counts = {}
            for word in words:
                counts[word] = counts.get(word, 0) + 1
            i_counts.append(counts)
        X_i = hasher.transform(i_counts)

        u_counts = []
        for words in cache["updates"]["lemmas stopped"].tolist():
            counts = {}
            for word in words:
                counts[word] = counts.get(word, 0) + 1
            u_counts.append(counts)

        X_u = hasher.transform(u_counts)

        K = cosine_similarity(X_i, X_u)
        for i, k_i in enumerate(K):
            feature_maps[i][self.ns()] = [(self.features()[0], 1 - k_i.max())]

        ##i#if len(sents) == len(df):
          #  cache[self.name()].append(fval)
          #  cache["X"] = scipy.sparse.vstack([cache["X"], X_d])


 

class BhattacharyyaCoefficient(FeatureGroup):
    def name(self):
        return "bc"

    def ns(self):
        return "b"

    def features(self):
        return ["bc"]

    def setup_cache(self, cache):
        cache[self.name()] = []
        cache["wc"] = defaultdict(int)

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):

        if cache["wc"]["__total__"] == 0:
            return 

        inp_counts = defaultdict(int)
        inp_tot = 0
        for words in df.iloc[sents]["lemmas stopped"].tolist():
            for word in words:
                inp_counts[word] += 1
                inp_tot += 1
        
        inp_tot = float(inp_tot)
        bc = 0.
        for word, count in inp_counts.iteritems():
            p_word = count / inp_tot
            q_word = cache["wc"][word] / cache["wc"]["__total__"]
            if q_word == 0: continue
            bc += np.sqrt(p_word * q_word)  #p_word * (np.log(p_word) - np.log(q_word))

        fval = bc
        if len(sents) == len(df):
            cache[self.name()].append(fval)

        next_f[self.ns()] = [(self.features()[0], fval)]


class KLDivergence(FeatureGroup):
    def name(self):
        return "kl_div"

    def ns(self):
        return "k"

    def features(self):
        return ["kl_div"]

    def setup_cache(self, cache):
        cache[self.name()] = []
        cache["wc"] = defaultdict(int)

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):

        if cache["wc"]["__total__"] == 0:
            return 

        inp_counts = defaultdict(int)
        inp_tot = 0
        for words in df.iloc[sents]["lemmas stopped"].tolist():
            for word in words:
                inp_counts[word] += 1
                inp_tot += 1
 
        inp_tot = float(inp_tot)
        kl = 0.
        v_size = len(set(inp_counts.keys()).union(set(cache["wc"].keys()))) - 1
        alp = .001
        for word, count in inp_counts.iteritems():
            p_word = count / inp_tot
            q_word = (cache["wc"][word] + alp * 1. / v_size)  / (cache["wc"]["__total__"] + alp)

            kl += p_word * (np.log(p_word) - np.log(q_word))

        fval = kl
        if len(sents) == len(df):
            cache[self.name()].append(fval)

        next_f[self.ns()] = [(self.features()[0], fval)]


class InputLexicalFeatures(FeatureGroup):
    def __init__(self, vocab):
        self.vocab = vocab

    def name(self):
        return "input-lex"

    def ns(self):
        return "i"

    def features(self):
        return self.vocab

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):
        tokens = [word for words in df.iloc[sents]["lemmas stopped"].tolist()
                  for word in words if word in self.vocab]
        tokens = list(set(tokens))
        next_f[self.ns()] = tokens
        
     

class UpdateLexicalFeatures(FeatureGroup):
    def __init__(self, vocab):
        self.vocab = vocab

    def name(self):
        return "update-lex"

    def ns(self):
        return "u"

    def features(self):
        return self.vocab

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):
        if cache is not None:
            tokens = [word for words in cache["lemmas stopped"].tolist()
                      for word in words if word in self.vocab]
        else:
            tokens = []
        tokens = list(set(tokens))
        select_f[self.ns()] = tokens
 
class UpdateInputLexicalFeatures(FeatureGroup):
    def __init__(self, vocab):
        self.vocab = vocab

    def name(self):
        return "update-input-lex"

    def ns(self):
        return "j"

    def features(self):
        return self.vocab

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):
        if cache is not None:
            upd_tokens = [
                word for words in cache["lemmas stopped"].tolist()
                for word in words]
            inp_tokens = [
                word for words in df.iloc[sents]["lemmas stopped"].tolist()
                for word in words]

            upd_per = []
            upd_loc = []
            upd_org = []
            upd_misc = []
            for upd_nes, upd_lems in izip(
                    cache["ne"].tolist(), cache["lemmas"].tolist()):
                for upd_ne, upd_lem in izip(upd_nes, upd_lems):
                    if upd_ne == "PERSON":
                        upd_per.append(upd_lem.lower())
                    elif upd_ne == "ORGANIZATION":
                        upd_org.append(upd_lem.lower())
                    elif upd_ne == "LOCATION":
                        upd_loc.append(upd_lem.lower())
                    elif upd_ne != "O":
                        upd_misc.append(upd_lem.lower())

            upd_per_set = set(upd_per)
            upd_loc_set = set(upd_loc)
            upd_org_set = set(upd_org)
            upd_misc_set = set(upd_misc)

            
            inp_per = []
            inp_loc = []
            inp_org = []
            inp_misc = []
            for inp_nes, inp_lems in izip(
                    df.iloc[sents]["ne"].tolist(), df.iloc[sents]["lemmas"].tolist()):
                for inp_ne, inp_lem in izip(inp_nes, inp_lems):
                    if inp_ne == "PERSON":
                        inp_per.append(inp_lem.lower())
                    elif inp_ne == "ORGANIZATION":
                        inp_org.append(inp_lem.lower())
                    elif inp_ne == "LOCATION":
                        inp_loc.append(inp_lem.lower())
                    elif inp_ne != "O":
                        inp_misc.append(inp_lem.lower())

            inp_per_set = set(inp_per)
            inp_loc_set = set(inp_loc)
            inp_org_set = set(inp_org)
            inp_misc_set = set(inp_misc)


            per_cov = len(upd_per_set.intersection(inp_per_set)) / \
                float(max(len(inp_per_set), 1))
            loc_cov = len(upd_loc_set.intersection(inp_loc_set)) / \
                float(max(len(inp_loc_set), 1))
            org_cov = len(upd_org_set.intersection(inp_org_set)) / \
                float(max(len(inp_org_set), 1))
            misc_cov = len(upd_misc_set.intersection(inp_misc_set)) / \
                float(max(len(inp_misc_set), 1))

            inp_tokens_set = set(inp_tokens)
            tokens = set(upd_tokens).intersection(inp_tokens_set)
            cov = len(tokens) / float(len(inp_tokens_set))            

            next_f[self.ns()] = [("coverage", cov ), 
                ("per_coverage", per_cov),
                ("loc_coverage", loc_cov),
                ("org_coverage", org_cov),
                ("misc_coverage", misc_cov),
            ]



class MarkovNextFeatures(FeatureGroup):
    def name(self):
        return "markov-next"

    def ns(self):
        return "m"
    
    def features(self):
        return ["q1_select", "q2_select", "q1_next", "q2_next", "<s1>", "<s2>",
                "selected_0", "selected_m1", "selected_m2", 
                "selected_m3", "selected_m4",
                "selected_m1_skip", "selected_m2_skip",
                "selected_m3_skip", "selected_m4_skip"]

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):
        idxs = [i for i in xrange(stream_idx - 2, stream_idx)]
        if idxs[0] < 0:
            q2 = ("<s2>", 1)
        else:
            q2 = ("q2_select", 1) if output[idxs[0]] == 1 else ("q2_next", 1)
        if idxs[1] < 0:
            q1 = ("<s1>", 1)
        else:
            q1 = ("q1_select", 1) if output[idxs[1]] == 1 else ("q1_next", 1)


        if stream_idx - 4 < 0: 
            qs4 = ("selected_m4", 0)
        else:
            qs4 = ("selected_m4", len(meta["sents_selected_in"][stream_idx-4]))
            if qs4[1] == 0:
                qs4 = ("selected_m4_skip", 1)
            else:
                qs4 = ("selected_m4", 1)

        if stream_idx - 3 < 0: 
            qs3 = ("selected_m3", 0)
        else:
            qs3 = ("selected_m3", len(meta["sents_selected_in"][stream_idx-3]))
            if qs3[1] == 0:
                qs3 = ("selected_m3_skip", 1)
            else:
                qs3 = ("selected_m3", 1)

        if stream_idx - 2 < 0: 
            qs2 = ("selected_m2", 0)
        else:
            qs2 = ("selected_m2", len(meta["sents_selected_in"][stream_idx-2]))
            if qs2[1] == 0:
                qs2 = ("selected_m2_skip", 1)
            else:
                qs2 = ("selected_m2", 1)

        if stream_idx - 1 < 0: 
            qs1 = ("selected_m1", 0)
        else:
            qs1 = ("selected_m1", len(meta["sents_selected_in"][stream_idx-1]))
            if qs1[1] == 0:
                qs1 = ("selected_m1_skip", 1)
            else:
                qs1 = ("selected_m1", 1)


        qs0 = ("selected_0", len(meta["sents_selected_in"][stream_idx]))

        #select_f[self.ns()] = [q1, q2]
        next_f[self.ns()] = [qs2, qs1, qs0] #[ q1, q2, qs2, qs1, qs0]
        #print [ qs2, qs1, qs0]

        

class InputUpdateSimpleSimilarityFeatures(FeatureGroup):

    def name(self):
        return "input-update-simple-sim"

    def ns(self):
        return "s"

    def features(self):
        feats = []
        for c in np.arange(0, 1, .2):
            feats.append(
                "cache2input_avg_sim_>{}".format(c))
            feats.append(
                "input2cache_avg_sim_>{}".format(c))
        
        return feats

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):

        if cache is not None:        
            select_f_list = []
            next_f_list = []        
            hasher = FeatureHasher()
            D_c = cache["lemmas stopped"].apply(lambda x: {k: 1 for k in x}).tolist()
            X_c = hasher.transform(D_c)
            D_d = df.iloc[sents]["lemmas stopped"].apply(lambda x: {k: 1 for k in x}).tolist()
            X_d = hasher.transform(D_d)
                
            K = cosine_similarity(X_c, X_d)
            for c in np.arange(0, 1, .2):
                K_gt_c = np.logical_and(K > c, c + .2 > K)
                next_f_list.append(
                    ("cache2input_avg_sim_>{}".format(c),
                     (np.sum(K_gt_c, axis=0) / float(K.shape[0])).mean()))
               # select_f_list.append(
               #     ("input2cache_avg_sim_>{}".format(c),
               #      np.sum(K_gt_c, axis=0).mean()))
            #select_f[self.ns()] = select_f_list
            next_f[self.ns()] = next_f_list #+ select_f_list



class UndirectedPageRank(FeatureGroup):
    def name(self):
        return "pr"

    def ns(self):
        return "p"

    def setup_cache(self, cache):
        cache["X"] = None
        cache["PR"] = None

    def features(self):
        return ["page_rank", "update_pr_effect"]    

    def make_SELECT_features(self, feature_maps, sents, df, cache, output,
            df_stream, stream_idx, meta):
   
        if len(sents) == len(df) or cache["PR"] == None:
            hasher = FeatureHasher()
            X_i = []
            for words in df["lemmas stopped"].tolist():
                counts = {}
                for word in words:
                    counts[word] = counts.get(word, 0) + 1
                X_i.append(counts)
            X_i = hasher.transform(X_i)

            cache["X"] = X_i

            if len(df) == 1:
                cache["PR"] = np.ones((1,1))
                
            else: 
                K = (X_i.dot(X_i.T) > 0).astype("int32").todense()
                
                degrees = K.sum(axis=1) - 1
                edges_x_2 = K.sum() - K.shape[0]
                if edges_x_2 == 0:
                    edges_x_2 = 1
                pr = 1. - degrees / float(edges_x_2)
                cache["PR"] = pr


        for i, s in enumerate(sents):
            feature_maps[i][self.ns()] = [(self.features()[0], cache["PR"][s,0])]

        if cache["updates"] is not None:
            hasher = FeatureHasher()
            X_u = []
            for words in cache["updates"]["lemmas stopped"].tolist():
                counts = {}
                for word in words:
                    counts[word] = counts.get(word, 0) + 1
                X_u.append(counts)
            X_u = hasher.transform(X_u)
            for i, s in enumerate(sents):
                X_pseud = scipy.sparse.vstack([cache["X"][s], X_u])
                K = (X_pseud.dot(X_pseud.T) > 0).astype("int32").todense()
                
                degrees = K.sum(axis=1) - 1
                edges_x_2 = K.sum() - K.shape[0]
                if edges_x_2 == 0:
                    edges_x_2 = 1
                pr = 1. - degrees / float(edges_x_2)
                feature_maps[i][self.ns()].append((self.features()[1], pr[0,0]))
        else:
            for i, s in enumerate(sents):
                feature_maps[i][self.ns()].append((self.features()[1], 1.))




class BasicFeatures(FeatureGroup):

    def name(self):
        return "basic"

    def ns(self):
        return "b"

    def setup_cache(self, cache):
        pass

    def cols(self):
        return [
#    "BASIC length", #"BASIC char length", 
    "BASIC doc position",]
#    "BASIC all caps ratio", "BASIC upper ratio",
#    "BASIC punc ratio",] # "BASIC person ratio", "BASIC organization ratio",
    #"BASIC date ratio", "BASIC time ratio", "BASIC duration ratio",
    #"BASIC number ratio", "BASIC ordinal ratio", "BASIC percent ratio",
    #"BASIC money ratio", "BASIC set ratio", "BASIC misc ratio"]

    def features(self):
        return [
#    "BASIC_length", #"BASIC_char_length", 
    "BASIC_doc_position",]
#    "BASIC_all_caps_ratio", "BASIC_upper_ratio", 
#    "BASIC_punc_ratio",]# "BASIC_person_ratio", "BASIC_organization_ratio",
#    "BASIC_date_ratio", "BASIC_time_ratio", "BASIC_duration_ratio",
#    "BASIC_number_ratio", "BASIC_ordinal_ratio", "BASIC_percent_ratio",
#    "BASIC_money_ratio", "BASIC_set_ratio", "BASIC_misc_ratio"]

    def make_SELECT_features(self, feature_maps, sents, df, cache, output,
            df_stream, stream_idx, meta):
        fnames = self.features()
        for r, fvals in enumerate(df.iloc[sents][self.cols()].values):
            feature_maps[r][self.ns()] = zip(fnames, fvals)
 
        #if stream_idx > 0:
        #    delta_avg_input =[(x[0], x[1] - hist[x[0]][:stream_idx].mean()) for x in avg_input]
        #    next_f[self.ns()] = delta_avg_input
        #else:
        #    delta_avg_input =[(x[0], x[1]) for x in avg_input]
            
        #if len(sents) == len(df):
        #    for f, v in avg_input:
        #        hist[f][stream_idx] = v

class NERatioFeatures(FeatureGroup):

    def name(self):
        return "ne_ratio"

    def ns(self):
        return "n"

    def setup_cache(self, cache):
        pass

    def cols(self):
        return[
            "BASIC person ratio", "BASIC organization ratio",
            "BASIC number ratio"]
    #"BASIC all caps ratio", "BASIC upper ratio", "BASIC lower ratio",
    #"BASIC punc ratio",] # "BASIC person ratio", "BASIC organization ratio",
    #"BASIC date ratio", "BASIC time ratio", "BASIC duration ratio",
    #"BASIC number ratio", "BASIC ordinal ratio", "BASIC percent ratio",
    #"BASIC money ratio", "BASIC set ratio", "BASIC misc ratio"]

    def features(self):
        return [
            "BASIC_person_ratio", "BASIC_organization_ratio",
            "BASIC_number_ratio"]

 #   "BASIC_all_caps_ratio", "BASIC_upper_ratio", "BASIC_lower_ratio",
 #   "BASIC_punc_ratio",]# "BASIC_person_ratio", "BASIC_organization_ratio",
#    "BASIC_date_ratio", "BASIC_time_ratio", "BASIC_duration_ratio",
#    "BASIC_number_ratio", "BASIC_ordinal_ratio", "BASIC_percent_ratio",
#    "BASIC_money_ratio", "BASIC_set_ratio", "BASIC_misc_ratio"]

    def make_SELECT_features(self, feature_maps, sents, df, cache, output,
            df_stream, stream_idx, meta):
        fnames = self.features()
        for r, fvals in enumerate(df.iloc[sents][self.cols()].values):
            feature_maps[r][self.ns()] = zip(fnames, fvals)
 




class LMFeatures(FeatureGroup):

    def name(self):
        return "lm"

    def ns(self):
        return "l"
    
    def setup_cache(self, cache):
        pass

    def cols(self):
        return ["LM domain avg lp", "LM gw avg lp"]

    def features(self):
        return ["LM_domain_avg_lp", "LM_gw_avg_lp"]

    def make_SELECT_features(self, feature_maps, sents, df, cache, output,
            df_stream, stream_idx, meta):
        fnames = self.features()
        for r, fvals in enumerate(df.iloc[sents][self.cols()].values):
            feature_maps[r][self.ns()] = [(fnames[0], fvals[0]), (fnames[1], fvals[1])]
        

class AveragedLMFeatures(FeatureGroup):

    def name(self):
        return "averaged-lm"

    def ns(self):
        return "l"

    def setup_cache(self, cache):
        cache[self.name()] = []
#    def features(self):
#        names = [x.replace(" ", "_") for x in self.cols()]
#        feats = [q + x for x in names for q in ["m1_", "m_2"]]
#        return feats

    def cols(self):
        return ["LM domain avg lp", "LM gw avg lp"]

    def features(self):
        return ["LM_domain_avg_lp", "LM_gw_avg_lp"]

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):

        feats = df.iloc[sents][self.cols()].mean().tolist()
        fnames = self.features()
        next_f[self.ns()] = [(fnames[0], feats[0]), (fnames[1], feats[1])]
        if len(sents) == len(df):
            cache[self.name()].append(feats)
#        feats = []
#        if stream_idx -1 >= 0:
#            df_m1 = df_stream[stream_idx-1]
#            non_updates = [
#                i for i in xrange(len(df_m1)) 
#                if i not in meta["sents_selected_in"][stream_idx - 1]]
#            if len(non_updates) > 0:
#                lm_m1 = [
#                    x for x in df_m1.iloc[non_updates][self.cols()].mean().iteritems()]
#                feats.extend([("m1_" + x[0], x[1]) for x in lm_m1])
#
#        if stream_idx -2 >= 0:
#            df_m2 = df_stream[stream_idx -2]
#            non_updates = [
#                i for i in xrange(len(df_m2)) 
#                if i not in meta["sents_selected_in"][stream_idx - 2]]
#            if len(non_updates) > 0:
#                lm_m2 = [
#                    x for x in df_m2.iloc[non_updates][self.cols()].mean().iteritems()]
#                feats.extend([("m2_" + x[0], x[1]) for x in lm_m2])
#        next_f[self.ns()] = feats
           

#        avg_input =  [(x[0].replace(" ", "_"), x[1])
#                      for x in df.iloc[sents][self.cols()].mean().iteritems()]
#        hist = cache[self.ns()]
#        if stream_idx > 0:
 #           delta_avg_input =[(x[0], x[1] - hist[x[0]][:stream_idx].mean()) for x in avg_input]
  #          next_f[self.ns()] = delta_avg_input
  #      else:
  #          delta_avg_input = avg_input #  [(x[0], x[1]) for x in avg_input]
  #      next_f[self.ns()] = avg_input
  #          
  #      if len(sents) == len(df):
   #         for f, v in avg_input:
   #             hist[f][stream_idx] = v

   #     next_f[self.ns()] = delta_avg_input

        #if cache is not None:
#            avg_upd =  [x for x in cache[self.features()].mean().iteritems()]
#            next_f[self.ns()] = avg_upd



class DocSize(FeatureGroup):

    def name(self):
        return "doc-size"

    def ns(self):
        return "z"

    def features(self):
        return ["doc-size"]

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):
        select_f[self.ns()] = len(sents)

class KLFeatures(FeatureGroup):
    def name(self):
        return "kl"

    def ns(self):
        return "k"

    def features(self):
        return ["kl_inp_upd", "kl_per"]

    def make_features(self, select_f, next_f, sents, df, cache, output,
            df_stream, stream_idx, meta):

        if cache["wc"]["__total__"] == 0:
            return 
                    
        inp_counts = defaultdict(int)
        inp_tot = 0
        for words in df.iloc[sents]["lemmas stopped"].tolist():
            for word in words:
                inp_counts[word] += 1
                inp_tot += 1
        
        inp_tot = float(inp_tot)
        kl = 0.
        for word, count in inp_counts.iteritems():
            p_word = count / inp_tot
            q_word = cache["wc"][word] / cache["wc"]["__total__"]
            if q_word == 0: continue
            kl += p_word * (np.log(p_word) - np.log(q_word))

        hist = cache[self.ns()] 
        if len(sents) == len(df):
            cache[self.ns()]["kl_inp_upd"][stream_idx] = kl
        
        per_counts = defaultdict(int)
        per_tot = 0
        for words, nes in izip(
                df.iloc[sents]["lemmas"].tolist(), df.iloc[sents]["ne"].tolist()):
            for word, ne in izip(words, nes):
                if ne == "PERSON":
                    per_counts[word] += 1
                    per_tot += 1
        
        per_tot = float(per_tot)
        kl_per = 0.
        for word, count in per_counts.iteritems():
            p_word = count / per_tot
            q_word = cache["per_wc"][word] / cache["per_wc"]["__total__"]
            if q_word == 0: continue
            kl_per += p_word * (np.log(p_word) - np.log(q_word))

        hist = cache[self.ns()] 
        if len(sents) == len(df):
            cache[self.ns()]["kl_per"][stream_idx] = kl
                
        next_f[self.ns()] = [
            ("kl_inp_upd", hist["kl_inp_upd"][:stream_idx].mean() - kl),
            ("kl_per", hist["kl_per"][:stream_idx].mean() - kl_per)]


def get_input_stream(event, extractor="goose", topk=20, delay=None, 
        threshold=.8):

    corpus = cuttsum.corpora.get_raw_corpus(event)
    res = InputStreamResource()
    return res.get_dataframes(
        event, corpus, extractor, threshold, delay, topk)



def main(features, training_ids, test_ids, sample_size, n_iters, report_dir):

    events = [e for e in cuttsum.events.get_events()
              if e.query_num in training_ids or e.query_num in test_ids]
    training_insts = []
    test_insts = []
    all_instances = []
    for event in events[1:]:
        print "Loading event", event.fs_name()
        corpus = cuttsum.corpora.get_raw_corpus(event)

        # A list of dataframes. Each dataframe is a document with =< 20 sentences.
        # This is the events document stream.
        dataframes = get_input_stream(event)
        
        if event.query_num in training_ids:
            training_insts.append((event, dataframes))    
           
        if event.query_num in test_ids:
            test_insts.append((event, dataframes))    
    
        all_instances.append((event, dataframes))

#    if "update-input-lex" in features \
#            or "update-lex" in features or "input-lex" in features:
#        print "Making vocab!"
#        vocab = make_vocab(training_insts)
#    else:
#        vocab = None
    vocab = None

#    print "Initializing feature groups and vw search task..."
#    vw = pyvw.vw(
#        "--search 0 --csoaa_ldf m --search_task hook --ring_size 1024 " + \
#        "--quiet  --search_no_caching")

#    vw = pyvw.vw(
#        "--search 2 --search_task hook --ring_size 1024 " + \
#        "--quiet  --search_no_caching")


#    dc_task = vw.init_search_task(DocumentClassifier)

    fgroups = [MaxInputUpdateSim(), UndirectedPageRank(), BasicFeatures()]



#    comm = MPI.COMM_WORLD 
#    status = MPI.Status()


#    idle_workers = []
#    n_workers = comm.size - 1

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    last = []
    for inst in training_insts:
        vw = pyvw.vw(
            "--search 0 --csoaa_ldf m --search_task hook --ring_size 1024 " + \
            "--quiet  --search_no_caching") # --interactions kblasm")


        dc_task = vw.init_search_task(DocumentClassifier)
        dc_task.register_features(fgroups)
        print inst[0].fs_name()
        ds_inst = downsample(inst, size=sample_size)
        for i in range(5):
            
            dc_task.learn([ds_inst])
            df = dc_task.score(ds_inst)
            path = os.path.join(
                report_dir, 
                "{}-ds-iter-{}.tsv".format(inst[0].fs_name(), i + 1))
            with open(path, "w") as f:
                df.to_csv(f, sep="\t", index=False)
            
            columns=["correct next", "tot next", "correct select", 
                     "tot select", "action", "oracle", "wgt acc", "acc",
                     "doc id", "nuggets", "oracle text", "pred text"] 
            
            f_cols = [col for col in df.columns if col.endswith("_w") or col.endswith("_v")]

            #for _, row in df.iterrows():
            print df[columns]
             #   for ns in dc_task.list_namespaces():
        
        last.append((inst[0], df[columns].tail(1)))
    for e, r in last:
        print e
        print r 
              #      print row[[x for x in f_cols if x.startswith(ns+"_")]].to_frame().T
              #  print
                
#    for i in range(10):
#        random.shuffle(training_insts)
#        for inst in training_insts:
#            ds_inst = downsample(inst)
#            dc_task.learn([ds_inst])
#        for inst in test_insts:
#            print dc_task.score(inst)
#            print i, inst[0].fs_name()
#    for n_iter in xrange(1, n_iters + 1):
#        print "iter {} ...".format(n_iter)
#
#        for training_id in training_ids:
#            sys.stdout.write("Sending {}\n".format(training_id))
#            data = comm.recv(
#                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
#            source = status.Get_source()
#            tag = status.Get_tag()
#            if tag == tags.READY:
#                comm.send([training_id, features, vocab, fgroup2fw],
#                    dest=source, tag=tags.WORKER_START)     
#
#        returned_fw = []
#        agg_fw = {}
#        for _ in training_ids:
#            new_fw = comm.recv(
#                source=MPI.ANY_SOURCE, tag=tags.SENDING_RESULT, status=status)
#            sys.stdout.write("Received data from {}\n".format(
#                status.Get_source()))
#            returned_fw.append(new_fw)
#
#        for new_fw in returned_fw:
#            aggregate_feature_weight_maps(agg_fw, new_fw)
#        average_aggregate_feature_weight_maps(agg_fw, len(training_ids))
#        fgroup2fw = agg_fw  
#        if n_iter > 100:
#            print "Running on test"
#            fgroups_dict = init_worker_vw(dc_task, features, fgroup2fw, vocab)
#            dc_task.register_features(fgroups_dict.values()) 
#            print "Done registering features"
#            for inst in training_insts:
#                print "eval on ", inst[0].fs_name() 
#                scores_df = dc_task.score(inst)
#                print scores_df
#                print inst[0].fs_name(), n_iter - 1
#                if not os.path.exists(report_dir):
#                    os.makedirs(report_dir)
#                path = os.path.join(
#                    report_dir, "{}_{}.tsv".format(n_iter - 1, inst[0].fs_name()))
#                with open(path, "w") as f:
#                    scores_df.to_csv(f, sep="\t", index=False)
#


    
#    fgroups_dict = init_worker_vw(dc_task, features, fgroup2fw, vocab)
#    dc_task.register_features(fgroups_dict.values()) 
#    for inst in training_insts:
#        scores_df = dc_task.score(inst)
#        print scores_df
#        print inst[0].fs_name(), n_iter
#        if not os.path.exists(report_dir):
#            os.makedirs(report_dir)
#        path = os.path.join(
#            report_dir, "{}_{}.tsv".format(n_iter, inst[0].fs_name()))
#        with open(path, "w") as f:
#            scores_df.to_csv(f, sep="\t", index=False)

#    for worker in xrange(n_workers):        
#        data = comm.recv(
#            source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
#        source = status.Get_source()
#        comm.send(None, dest=source, tag=tags.WORKER_STOP)     

def aggregate_feature_weight_maps(agg_fw, new_fw):
    for fname, fw in new_fw.items():
        if fname in agg_fw:
            agg_fname_fw = agg_fw[fname]
            for k, v in fw.items():
                agg_fname_fw[k] += v 
        else:
            agg_fw[fname] = fw

def average_aggregate_feature_weight_maps(agg_fw, Z):
    Z = float(Z)
    for fw in agg_fw.values():
        for k in fw.keys():
            fw[k] /= Z
        

def worker(sample_size):

    events = cuttsum.events.get_events()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    status = MPI.Status()
    instances = {}
    
#    vw = pyvw.vw(
#        "--search 0 --csoaa_ldf m --search_task hook --ring_size 1024 " + \
#        "--quiet  --search_no_caching")

    vw = pyvw.vw(
        "--search 2 --search_task hook --ring_size 1024 " + \
        "--quiet  --search_no_caching")

    dc_task = vw.init_search_task(DocumentClassifier)

    while True: 
    
        comm.send(None, dest=0, tag=tags.READY)
        data = comm.recv(
            source=0, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.WORKER_START:
            training_id, features, vocab, fgroup2fw = data
            msg = "worker-{} running on event {} with features:\n"
            msg += "\n".join("\t" + fname for fname in features)
            sys.stdout.write(
                msg.format(rank, training_id) + "\n")

            if training_id not in instances:
                event = [e for e in events if e.query_num == training_id][0]
                sys.stdout.write(
                    "worker-{}: loading event {}\n".format(
                        rank, event.fs_name()))
                dataframes = get_input_stream(event)
                instances[training_id] = (event, dataframes)

            fgroups = init_worker_vw(dc_task, features, fgroup2fw, vocab)
            dc_task.register_features(fgroups.values())
            instance = instances[training_id]
            ds_inst = downsample(instance, sample_size)  
             
            for i in range(10):
                dc_task.learn([ds_inst])               
                print dc_task.score(instance)
                print i

            fw = {}
            for fname, fgroup in fgroups.items():
                fw[fname] = fgroup.get_weights(dc_task)

            sys.stdout.write("worker-{} sending result for event {}\n".format(
                rank, event.fs_name()))
            comm.send(fw, dest=0, tag=tags.SENDING_RESULT)
            sys.stdout.write("worker-{} sent result for event {}\n".format(
                rank, event.fs_name()))
            
        
        elif tag == tags.WORKER_STOP:
            break

    sys.stdout.write("worker-{} shutting down!\n".format(rank))

 
def init_worker_vw(search_task, features, fgroup2fw, vocab):
    fgroups = {}
    for fname in features:
        if fname == "update-lex":
            fgroup = UpdateLexicalFeatures(vocab)
            fgroup.initialize_vw(search_task)
            fgroup.set_weights(search_task, fgroup2fw[fgroup.name()])            
            fgroups[fgroup.name()] = fgroup
        elif fname == "input-lex":
            fgroup = InputLexicalFeatures(vocab)
            fgroup.initialize_vw(search_task)
            fgroup.set_weights(search_task, fgroup2fw[fgroup.name()])            
            fgroups[fgroup.name()] = fgroup
        elif fname == "update-input-lex":
            fgroup = UpdateInputLexicalFeatures(vocab)
            fgroup.initialize_vw(search_task)
            fgroup.set_weights(search_task, fgroup2fw[fgroup.name()])            
            fgroups[fgroup.name()] = fgroup
        elif fname == "markov-next":
            fgroup = MarkovNextFeatures()
            fgroup.initialize_vw(search_task)
            fgroup.set_weights(search_task, fgroup2fw[fgroup.name()])            
            fgroups[fgroup.name()] = fgroup
        elif fname == "input-update-simple-sim":
            fgroup = InputUpdateSimpleSimilarityFeatures()
            fgroup.initialize_vw(search_task)
            fgroup.set_weights(search_task, fgroup2fw[fgroup.name()])            
            fgroups[fgroup.name()] = fgroup
        elif fname == "averaged-basic":
            fgroup = AveragedBasicFeatures()
            fgroup.initialize_vw(search_task)
            fgroup.set_weights(search_task, fgroup2fw[fgroup.name()])            
            fgroups[fgroup.name()] = fgroup
        elif fname == "averaged-lm":
            fgroup = AveragedLMFeatures()
            fgroup.initialize_vw(search_task)
            fgroup.set_weights(search_task, fgroup2fw[fgroup.name()])            
            fgroups[fgroup.name()] = fgroup

        elif fname == "doc-size":
            fgroup = DocSize()
            fgroup.initialize_vw(search_task)
            fgroup.set_weights(search_task, fgroup2fw[fgroup.name()])            
            fgroups[fgroup.name()] = fgroup



       

    return fgroups


def make_vocab(instances):
    word_counts = defaultdict(int)
    for event, dataframes in instances:
        event_vocab = set(
            [word 
             for df in dataframes
             for words in df["lemmas stopped"].tolist()
             for word in words])
        for word in event_vocab:
            word_counts[word] += 1
    vocab = set([word for word, count in word_counts.items() if count >= 3])
    return vocab

def downsample(training_instance, size=90):
    ds_insts = []
    event, dataframes = training_instance
    arange = [x for x in xrange(len(dataframes))]
    random.shuffle(arange)
    arange = arange[:size]
    arange.sort()
    #print "downsampling", event.fs_name()
    #print arange
    ds_df = [dataframes[i] for i in arange]
    return (event, ds_df)



if __name__ == "__main__":

    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument(
        u"--features", nargs=u"+",
        choices=["markov-next", "update-input-lex", "input-lex", "update-lex",
            "input-update-simple-sim", "averaged-basic", "averaged-lm", "doc-size"],
        help=u"features to use")
    parser.add_argument(u"--training-event-ids", type=int, nargs=u"+",
                        help=u"event ids to select.")
    parser.add_argument(u"--test-event-ids", type=int, nargs=u"+",
                        help=u"event ids to select.")
    parser.add_argument(u"--report-dir",
                        help=u"Where to right outputs.")
    parser.add_argument(u"--num-iters", type=int,
                        help=u"training iters")
    parser.add_argument(u"--sample-size", type=int, default=90,
                        help=u"down sample size")

    args = parser.parse_args()

    n_iters = args.num_iters
    report_dir = args.report_dir
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    training_ids = args.training_event_ids    
    test_ids = args.test_event_ids
    features = args.features
    sample_size = args.sample_size

    #rank = MPI.COMM_WORLD.rank
    #size = MPI.COMM_WORLD.size
    
   # if size <= 1:
   #     print "Need at least 2 processes!"
   #     exit()
    
    #if rank == 0:
    main(features, training_ids, test_ids, sample_size, n_iters, report_dir)
    #else:
    #    worker(sample_size)
