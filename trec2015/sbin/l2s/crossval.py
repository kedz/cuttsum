import cuttsum.events
import cuttsum.corpora
from cuttsum.pipeline import InputStreamResource
from mpi4py import MPI
from cuttsum.misc import enum
from cuttsum.classifiers import NuggetRegressor
import numpy as np
import pandas as pd
import random
import pyvw
from datetime import datetime
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics.pairwise import cosine_similarity
from itertools import izip
import os
import cuttsum.judgements
from cuttsum.misc import event2semsim
import math

np.random.seed(42)

matches_df = cuttsum.judgements.get_merged_dataframe()
tags = enum("READY", "WORKER_START", "WORKER_STOP")

class FeatureMapper(dict):
    def __init__(self):
        self.store = dict()
        self._inv_store = dict()
        self._idx = 0
    def __getitem__(self, key):
        if key not in self.store:
            self.store[key] = self._idx
            self._inv_store[self._idx] = key
            self._idx += 1
        return self.store[key]
    def items(self):
        return self.store.items()
    def inv(self, idx):
        return self._inv_store[idx]
    

fmap = FeatureMapper()

SELECT = 1
SKIP = 2

basic_cols = ["BASIC length", #"BASIC char length",
    "BASIC doc position", "BASIC all caps ratio",
    "BASIC upper ratio", 
#    "BASIC lower ratio",
#    "BASIC punc ratio", 
    "BASIC person ratio",
    "BASIC location ratio",
    "BASIC organization ratio", "BASIC date ratio",
    "BASIC time ratio", "BASIC duration ratio",
    "BASIC number ratio", "BASIC ordinal ratio",
    "BASIC percent ratio", 
    "BASIC money ratio",
#    "BASIC set ratio", "BASIC misc ratio"
]

query_bw_cols = [
    "Q_sent_query_cov",
    "Q_sent_syn_cov",
    "Q_sent_hyper_cov",
    "Q_sent_hypo_cov",
]

query_fw_cols = [
    "Q_query_sent_cov",
    "Q_syn_sent_cov",
    "Q_hyper_sent_cov",
    "Q_hypo_sent_cov",
]

lm_cols = ["LM domain avg lp",
           "LM gw avg lp"]

sum_cols = [
    "SUM_sbasic_sum",
    "SUM_sbasic_amean",
#    "SUM_sbasic_max",
    "SUM_novelty_gmean",
    "SUM_novelty_amean",
#    "SUM_novelty_max",
    "SUM_centrality",
    "SUM_pagerank",
    "SUM_sem_novelty_gmean",
    "SUM_sem_novelty_amean",
    "SUM_sem_centrality",
    "SUM_sem_pagerank",

]

stream_cols = [
    "STREAM_sbasic_sum",
    "STREAM_sbasic_amean",
    "STREAM_sbasic_max",
    "STREAM_per_prob_sum",
    "STREAM_per_prob_max",
    "STREAM_per_prob_amean",
    "STREAM_loc_prob_sum",
    "STREAM_loc_prob_max",
    "STREAM_loc_prob_amean",
    "STREAM_org_prob_sum",
    "STREAM_org_prob_max",
    "STREAM_org_prob_amean",
    "STREAM_nt_prob_sum",
    "STREAM_nt_prob_max",
    "STREAM_nt_prob_amean",
]

best_feats = set([
    "LM gw avg lp",
    "CACHE_SIM_amean",
    "STREAM_sbasic_max",
    "CACHE_SIM_max",
    "STREAM_loc_prob_amean^df^prob",
    "BASIC ordinal ratio^df^prob",
    "STREAM_per_prob_max",
    "BASIC ordinal ratio",
    "STREAM_sbasic_max^df",
    "SUM_pagerank",
    "BASIC date ratio^df^prob",
    "SUM_sem_centrality^prob",
    "BASIC doc position^df",
    "STREAM_org_prob_amean",
    "STREAM_nt_prob_sum^df",
    "STREAM_loc_prob_sum",
    "STREAM_loc_prob_max^df",
    "STREAM_nt_prob_sum^prob",
    "BASIC location ratio^df",
    "BASIC all caps ratio^prob",
    "BASIC organization ratio^df^prob",
    "SUM_sbasic_sum^df",
    "STREAM_org_prob_sum^prob",
    "BASIC money ratio^prob",
    "CONSTANT",
    "STREAM_loc_prob_max",
    "STREAM_org_prob_amean^prob",
    "STREAM_nt_prob_max^prob",
    "SUM_sbasic_sum^df^prob",
    "STREAM_nt_prob_sum",
    "LM domain avg lp^df",
    "BASIC number ratio^df^prob",
    "CACHE_SEM_SIM_amean",
    "Q_syn_sent_cov",
    "BASIC percent ratio",
    "BASIC time ratio^df^prob",
    "BASIC date ratio^prob",
    "BASIC person ratio^prob",
    "STREAM_sbasic_sum^df",
    "BASIC location ratio^df^prob",
    "BASIC money ratio",
    "BASIC duration ratio^df^prob",
    "BASIC location ratio^prob",
    "BASIC duration ratio^prob",
    "BASIC person ratio^df",
    "STREAM_sbasic_amean",
    "BASIC date ratio",
    "SUM_sem_centrality^df",
    "BASIC time ratio^df",
    "STREAM_sbasic_sum",
])


class Summarizer(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw , sch, num_actions)
        #sch.set_options( sch.AUTO_HAMMING_LOSS )
        self._with_scores = False
        self._loss = 1
        self.total_loss = 0
        self.log_time = False
        self.use_best_feats = False
        self.use_i_only = False
        self.use_abs_df = False
        self._scores = []
        self._keep_scores = False
        self._doc_condition = False
    
    def feat_weight(self, idx):
        if idx >= fmap._idx:
            ridx = idx - fmap._idx
        else:
            ridx = idx
        ex = self.example({"a": [(idx, 1)]})
        w = self.vw.get_weight(ex.feature("a", 0))
        return fmap.inv(ridx), idx, w
    
    def get_feature_weights(self):
        select_feats = []
        for i in xrange(fmap._idx):
            fname, idx, weight = self.feat_weight(i)
            select_feats.append({"name": fname, "index": idx, "weight": weight})
        select_feats.sort(key=lambda x: x["weight"])
        select_df = pd.DataFrame(select_feats, columns=["name", "index", "weight"])
        next_feats = []
        for i in xrange(fmap._idx, fmap._idx * 2):
            fname, idx, weight = self.feat_weight(i)
            next_feats.append({"name": fname, "index": idx, "weight": weight})
        next_feats.sort(key=lambda x: x["weight"])
        next_df = pd.DataFrame(next_feats, columns=["name", "index", "weight"])
        return select_df, next_df

    def set_weights(self, weights_df):
        for _, row in weights_df.iterrows():
            idx = row["index"]
            weight = row["weight"]
            self.vw.set_weight(idx, 0, weight)
        
  #for i, feat in enumerate(self.basic_cols()):
            #fw.append(("b:" + feat, w))
        #fw.append(("n:bias", self.vw.get_weight(ex.feature("n", 0))))


    def set_loss(self, loss):
        self._loss = loss

    def make_example(self, sent, cache, cache_in, days, x, cache_latent, dfdelta, ):
        if self.log_time is True:
            days = np.log(2 + days)
        df = sent.to_frame().transpose()

        if self._doc_condition is True:
            if cache is not None:
                doc_condition = (cache["stream id"] == sent["stream id"]).astype("int32").sum()
                dc_feat = [("NUM_PREV_SELECTED_IN_DOC_{}".format(doc_condition), 1)]
            else:
                dc_feat = [("NUM_PREV_SELECTED_IN_DOC_0", 1)]

        else: dc_feat = []

        feats = {
            "b": [(k, df[k].tolist()[0]) for k in basic_cols],
            "c": [(k, df[k].tolist()[0]) for k in sum_cols],
            "q": [(k, df[k].tolist()[0]) for k in query_fw_cols],
            "l": [(k, df[k].tolist()[0]) for k in lm_cols],
            "s": [(k, df[k].tolist()[0]) for k in stream_cols],
            "p": [("probs",sent["probs"])],
            "d": dc_feat,
            #"t": [("time", days)],
         #   "g": ["GAIN_POS" if gain > 0 else "GAIN_ZERO"],
            
        }  
           

        if cache is None:
            feats["g"] = [("EMPTY", 1)]

        else:
            h = FeatureHasher(input_type="string")
            X_c = h.transform(cache["lemmas stopped"].tolist())
            x_i = h.transform([sent["lemmas stopped"]])
            K = cosine_similarity(X_c, x_i)
            k_u = K.mean()
            k_max = K.max()
            if k_max == 0:
                feats["g"] = [("CACHE_SIM_ZERO", 1)]
            else:
                feats["g"] = [("CACHE_SIM_amean", k_u), ("CACHE_SIM_max", k_max)]
        
            K_l = cosine_similarity(cache_latent, x)
            k_lu = K_l.mean()
            k_lmax = K_l.max()
            if k_lmax == 0:
                feats["g"].append(("CACHE_SEM_SIM_ZERO", 1))
            else:        
                feats["g"].extend([("CACHE_SEM_SIM_amean", k_lu), ("CACHE_SEM_SIM_max", k_lmax)])

        feats["I"] = []
        
#        for ns in ["b", "c", "q", "l", "s", "g"]:
#            for feat, val in feats[ns]:
#                feats["I"].append(("{}^time^prob".format(feat), val * sent["probs"] * days))
#                feats["I"].append(("{}^prob".format(feat), val * sent["probs"]))
#                feats["I"].append(("{}^time".format(feat), val * days))

        for ns in ["b", "c", "q", "l", "s", "g", "d"]:
            for feat, val in feats[ns]:
                feats["I"].append(("{}^df^prob".format(feat), val * sent["probs"] * dfdelta))
                feats["I"].append(("{}^prob".format(feat), val * sent["probs"]))
                feats["I"].append(("{}^df".format(feat), val * dfdelta))
 
 
        ifeats = {'a': []}
        if self.use_i_only:
            NS = ["I"]
        else:
            NS = ["b", "c", "q", "l", "s", "g", "p", "d", "I"]

        if self.use_best_feats:

            for ns in NS:
                for key, val in feats[ns]: 
                    if key not in best_feats: continue
                    ifeats['a'].append((fmap[key], val))
            ifeats['a'].append((fmap["CONSTANT"], 1))

        else:
            for ns in NS:
                for key, val in feats[ns]: 
                    ifeats['a'].append((fmap[key], val))
            ifeats['a'].append((fmap["CONSTANT"], 1))

        ex = self.example(ifeats)
                 
            
            #select_weight = sum(self.vw.get_weight(idx, 0) * val for idx, val in ifeats['a'])
            #next_weight = sum(self.vw.get_weight(fmap._idx + idx , 0) * val for idx, val in ifeats['a'])
            #print select_weight, next_weight            
#self._scores.append({"SELECT": select_weight, "NEXT": next_weight})            
         #   self._scores.append(ex.get_partial_prediction())
            #print select_weight, next_weight, self.example(ifeats).get_label() #, self.example(ifeats).get_costsensitive_partial_prediction(1)  #self.vw.get_partial_prediction(self.example(ifeats)
             
        
        return ex

    def _run(self, (event, df_stream, X_stream, dfdeltas)):

        nuggets = set()
        cache = None
        cache_in = None 
        cache_latent = None
        output = []
        current_dfdelta_idx = 0
        current_dfdelta = 0        

        n = 0
        y_int_y_hat = 0
        size_y = 0
        size_y_hat = 0

        loss = 0

        for (_, sent), x in zip(df_stream.iterrows(), X_stream):
            intts = int(sent["timestamp"])
            if intts > dfdeltas[current_dfdelta_idx +1][0]:
                current_dfdelta_idx += 1
                current_dfdelta = dfdeltas[current_dfdelta_idx][1]
                if self.use_abs_df:
                    current_dfdelta = abs(current_dfdelta)

            days = (datetime.utcfromtimestamp(int(sent["timestamp"])) - event.start).total_seconds() / (60. * 60. * 24.)
            n += 1
            gain = len(sent["nuggets"] - nuggets)
            if self.sch.predict_needs_example():
                examples = self.make_example(sent, cache, cache_in, days, x, cache_latent, current_dfdelta)
            else:
                examples = None
            if gain > 0:
                oracle = SELECT
            else:
                oracle = SKIP

            # Make prediction.
            pred = self.sch.predict(
                examples=examples,
                my_tag=n,
                oracle=oracle,
                condition=[], # (n-1, "p"), ])
            )
            output.append(pred)

            #if examples is not None:
            if self._keep_scores:
            #if self._keep_scores:
                #print "HERE"
                #lab = pyvw.cost_sensitive_label()
                #print "HERE2"
                #lab.from_example(examples)
                #print "HERE3"
                #for wc in lab.costs:
                 #   print wc.partial_prediction,
                #print pred
                #print examples.get_costsensitive_partial_prediction(1)
            #select_weight = sum(self.vw.get_weight(idx, 0) * val for idx, val in ifeats['a'])
            #next_weight = sum(self.vw.get_weight(fmap._idx + idx , 0) * val for idx, val in ifeats['a'])
            #self._scores.append({"SELECT": select_weight, "NEXT": next_weight})            
                self._scores.append(examples.get_partial_prediction())

            #if examples is not None: 
            #    print self._scores[-1], examples.get_partial_prediction(), "SELECT" if oracle == SELECT else "NEXT", "PRED: SELECT" if pred == SELECT else "PRED: NEXT"

                #print examples.get_simplelabel_label(), examples.get_multiclass_label(), oracle, 1 / (1 + math.exp(-examples.get_partial_prediction())), pred
                #print pyvw.cost_sensitive_label().from_example(examples)
               # if self._keep_scores:
               #     ascore = self._scores[-1]
               #     print ascore, ascore["SELECT"] >= ascore["NEXT"] if pred == SELECT else ascore["SELECT"] <= ascore["NEXT"], ascore["SELECT"] + ascore["NEXT"]
            #print select_weight, next_weight, self.example(ifeats).get_label() #, self.example(ifeats).get_costsensitive_partial_prediction(1)  #self.vw.get_partial_prediction(self.example(ifeats)
            if pred != oracle:
                if oracle == SELECT:
                    loss += self._loss
                else:
                    loss += 1

            if pred == SELECT and oracle == SELECT:
                y_int_y_hat += 1
                size_y += 1
                size_y_hat += 1
            elif pred == SELECT and oracle == SKIP:
                size_y_hat += 1

            elif pred == SKIP and oracle == SELECT:
                size_y += 1



            #if self._with_scores is True:
            #    print "examining:", sent["pretty text"]        

            if pred == SELECT:
                nuggets.update(sent["nuggets"])
                if cache is None:
                    cache = sent.to_frame().transpose()
                    cache_latent = x
                else:
                    cache = pd.concat([cache, sent.to_frame().transpose()])
                    cache_latent = np.vstack([cache_latent, x])
#            else:
#                if cache_in is None:
#                    cache_in = sent.to_frame().transpose()
#                else:
#                    cache_in = pd.concat([cache_in, sent.to_frame().transpose()])

        Z = size_y + size_y_hat
        if Z == 0: Z = 1
        loss = 1 - float(y_int_y_hat) / Z

        self.sch.loss(loss)

        self.total_loss += loss
        return output





def start_manager(event_ids, output_dir):

    events = [e for e in cuttsum.events.get_events()
              if e.query_num in event_ids]

    jobs = [(events[0:i] + events[i+1:], test_event)
            for i, test_event in enumerate(events)]


    comm = MPI.COMM_WORLD #.Accept(port, info, 0)
    status = MPI.Status() # get MPI status object

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    w_path = os.path.join(output_dir, "weights.tsv")
    s_path = os.path.join(output_dir, "scores.tsv")
    t_path = os.path.join(output_dir, "summary.tsv")

    n_workers = comm.size - 1
    
    first_write = True

    with open(w_path, "w") as w_f, open(s_path, "w") as s_f, \
            open(t_path, "w") as t_f:
        while n_workers > 0:
            data = comm.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            source = status.Get_source()
            tag = status.Get_tag()
            print "STATUS", tag, "SOURCE", source
            if tag == tags.READY:
                if len(jobs) > 0:
                    job = jobs.pop(0)
                    comm.send(job, dest=source, tag=tags.WORKER_START)
                else:
                    comm.send(None, dest=source, tag=tags.WORKER_STOP)
                    n_workers -= 1
                if data is not None:
                    scores_df, weights_df, summary_df = data
                    scores_df.to_csv(s_f, sep="\t", index=False, header=first_write)
                    s_f.flush()
                    weights_df.to_csv(w_f, sep="\t", index=False, header=first_write)
                    w_f.flush()
                    summary_df.to_csv(t_f, sep="\t", index=False, header=first_write)
                    t_f.flush()
                    first_write = False


def get_all_semsim():
    accident_semsim = event2semsim("accident")
    natdis_semsim = event2semsim("earthquake")
    social_semsim = event2semsim("protest")
    terror_semsim = event2semsim("shooting")
    return {
        "accident": accident_semsim,
        "earthquake": natdis_semsim,
        "storm": natdis_semsim,
        "impact event": natdis_semsim,
        "shooting": terror_semsim,                
        "hostage": terror_semsim,                
        "conflict": terror_semsim,                
        "bombing": terror_semsim,                
        "protest": social_semsim,
        "riot": social_semsim,
    }     

def get_dfdeltas():
    with open("doc_freqs.tsv", "r") as f:
        df = pd.read_csv(f, sep="\t")
    def get(event):
        df_e = df[df["event"] == event.query_id]
        mylist = [[0,0]] + zip(df_e["hour"].tolist(), df_e["df delta"].tolist())
        return mylist
    return get

def start_worker(sample_size, samples_per_event, gold_probs, iters, l2, log_time,
        use_best_feats, use_i_only, use_abs_df):
    rank = MPI.COMM_WORLD.Get_rank()
    status = MPI.Status() # get MPI status object
    job_results = None
    semsims = get_all_semsim()
    dfdeltas = get_dfdeltas()


    while True:
        comm.send(job_results, dest=0, tag=tags.READY)
        data = comm.recv(
            source=0, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.WORKER_START:
            training_events, test_event = data
            print "JOBBING", test_event.fs_name()
            job_results = do_work(
                training_events, test_event, 
                sample_size, samples_per_event, gold_probs, iters, l2, log_time, semsims, dfdeltas,
                use_best_feats, use_i_only, use_abs_df)

        if tag == tags.WORKER_STOP:
            break
    
 
def do_work(training_events, test_event, sample_size, samples_per_event,
        gold_probs, iters, l2, log_time, semsims, dfdeltas,
        use_best_feats, use_i_only, use_abs_df):
    
    training_streams = []
    summary = []

    for event in training_events:
        df = get_input_stream(event, gold_probs)
        training_streams.append((event, df))

    test_df = get_input_stream(test_event, gold_probs)
    test_X_l = semsims[test_event.type].transform(
        test_df["stems"].apply(lambda x: ' '.join(x)).tolist())
    test_stream = (test_event, test_df, test_X_l, dfdeltas(test_event))

    vw = pyvw.vw(
        ("--l2 {} --search 2 --search_task hook --ring_size 1024 " + \
         "--search_no_caching --noconstant --quiet").format(l2)) 
    task = vw.init_search_task(Summarizer)
    task.use_best_feats = use_best_feats
    task.use_i_only = use_i_only
    task.use_abs_df = use_abs_df
    print "use best?", task.use_best_feats 
    print "use i only?", task.use_i_only
    print "use abs df?", task.use_abs_df
    task.log_time = log_time
    all_scores = []
    all_weights = []

    instances = []
    for sample in xrange(samples_per_event):
        for event, stream in training_streams:
            while 1:
                sample_stream = ds(stream, sample_size=sample_size)
                if (sample_stream["nuggets"].apply(len) > 0).any():
                    break
            X_l = semsims[event.type].transform(
                sample_stream["stems"].apply(lambda x: ' '.join(x)).tolist())  
            instances.append((event, sample_stream, X_l, dfdeltas(event)))



    for n_iter in xrange(1, iters + 1):
        task.total_loss = 0    

        
        #instances = [(event, ds(stream, sample_size=sample_size))
        #             for event, stream in training_streams
        #             for sample in xrange(samples_per_event)]
        random.shuffle(instances)
        for i, inst in enumerate(instances):
            print "{}.{}.{}/{}".format(
                test_event.fs_name(), n_iter, i, len(instances))
            task.learn([inst])
        print "{}.{}.p".format(
            test_event.fs_name(), n_iter)
        
        train_egain = 0
        train_comp = 0
        train_f1 = 0
        train_loss = 0
        for i, inst in enumerate(instances):
            egain, comp, f1, loss, train_sum = predict(task, inst, n_iter)
            train_egain += egain
            train_comp += comp
            train_f1 += f1
            train_loss += loss
        train_egain = train_egain / float(len(instances))
        train_comp = train_comp / float(len(instances))
        train_f1 = train_f1 / float(len(instances))
        train_loss = train_loss / float(len(instances))
        print "{} {} train loss {}".format(test_event.query_id, n_iter, train_loss)


        pred = task.predict(test_stream)

        select_df, next_df = task.get_feature_weights()

        select_df["class"] = "SELECT"
        select_df["iter"] = n_iter

        next_df["class"] = "NEXT"
        next_df["iter"] = n_iter
        all_weights.append(select_df)
        all_weights.append(next_df)

        pred = ["SELECT" if p == SELECT else "SKIP" for p in pred]
        all_nuggets = set()
        for nuggets in test_stream[1]["nuggets"].tolist():
            all_nuggets.update(nuggets)

        loss = 0        
        y_int_y_hat = 0
        size_y = 0
        size_y_hat = 0

        nuggets = set()
        for action, (_, sent) in izip(pred, test_stream[1].iterrows()):
            gain = len(sent["nuggets"] - nuggets)
            if action == "SELECT": 
                if gain == 0:
                    loss += 1
                summary.append({
                    "event": test_event.query_id,
                    "iter": n_iter,
                    "update id": sent["update id"],
                    "timestamp": sent["timestamp"],
                    "gain": gain, 
                    "nuggets": ",".join(sent["nuggets"]), 
                    "update text": sent["pretty text"]
                })
                nuggets.update(sent["nuggets"])
            else: 
                if gain > 0:
                    loss += 1
            if gain > 0:
                oracle = "SELECT"
            else:
                oracle = "SKIP"

            if action == "SELECT" and oracle == "SELECT":
                y_int_y_hat += 1
                size_y += 1
                size_y_hat += 1
            elif action == "SELECT" and oracle == "SKIP":
                size_y_hat += 1
            elif action == "SKIP" and oracle == "SELECT":
                size_y += 1


        if size_y_hat == 0:
            print test_event
            print (test_stream[1]["nuggets"].apply(len) > 0).any()
        loss = 1 - float(y_int_y_hat) / (size_y + size_y_hat)
        

        if len(nuggets) > 0:
            egain = len(nuggets) / sum([1.0 if a == "SELECT" else 0.0 for a in pred])
        else:
            egain = 0        
        comp = len(nuggets) / float(len(all_nuggets)) 
        
        all_scores.append({"iter": n_iter, "Comp.": comp,
                           "E[gain]": egain, "Loss": loss, 
                           "Avg. Train Loss": train_loss,
                           "Avg. Train E[gain]": train_egain,
                           "Avg. Train Comp.": train_comp,
                           "Avg. Train F1": train_f1,
        })        

        print "{}.{}.p E[gain]={:0.6f} Comp.={:0.6f} Train Loss={:0.6f}".format(
            test_event.fs_name(), n_iter, egain, comp, train_loss)

    scores_df = pd.DataFrame(all_scores, columns=["iter", "E[gain]", "Comp.", "Loss", "Avg. Train Loss", "Avg. Train E[gain]", "Avg. Train Comp.", "Avg. Train F1"])
    weights_df = pd.concat(all_weights)
    weights_df["event"] = test_event.query_id
    scores_df["event"] = test_event.query_id
    summary_df = pd.DataFrame(
        summary, 
        columns=["iter", "event", "update id", "timestamp", "gain", 
                 "update text", "nuggets"])
    return scores_df, weights_df, summary_df


def get_input_stream(event, gold_probs, extractor="goose", thresh=.8, delay=None, topk=20, 
        use_2015F=False, truncate=False):
    max_nuggets = 3
    
    corpus = cuttsum.corpora.get_raw_corpus(event)
    if use_2015F is True and event.query_num > 25:
        corpus = cuttsum.corpora.FilteredTS2015()
    print event, corpus

    res = InputStreamResource()
    df = pd.concat(
        res.get_dataframes(event, corpus, extractor, thresh, delay, topk))

    selector = (df["n conf"] == 1) & (df["nugget probs"].apply(len) == 0)
    df.loc[selector, "nugget probs"] = df.loc[selector, "nuggets"].apply(lambda x: {n:1 for n in x})

    df["true probs"] = df["nugget probs"].apply(lambda x: [val for key, val in x.items()] +[0])
    df["true probs"] = df["true probs"].apply(lambda x: np.max(x))
    df.loc[(df["n conf"] == 1) & (df["nuggets"].apply(len) == 0), "true probs"] = 0

    if gold_probs is True:
        df["probs"] = df["true probs"]
    else:
        df["probs"] = NuggetRegressor().predict(event, df)   
    
    df["nuggets"] = df["nugget probs"].apply(
        lambda x: set([key for key, val in x.items() if val > .9]))


    nid2time = {}
    nids = set(matches_df[matches_df["query id"] == event.query_id]["nugget id"].tolist())
    for nid in nids:
        ts = matches_df[matches_df["nugget id"] == nid]["update id"].apply(lambda x: int(x.split("-")[0])).tolist()
        ts.sort()
        nid2time[nid] = ts[0]

    fltr_nuggets = []
    for name, row in df.iterrows():
        fltr_nuggets.append(
            set([nug for nug in row["nuggets"] if nid2time[nug] <= row["timestamp"]]))
    #print df[["nuggets", "timestamp"]].apply(lambda y: print y[0]) #  datetime.utcfromtimestamp(int(y["timestamp"])))
    #print nids
    df["nuggets"] = fltr_nuggets

    df["nuggets"] = df["nuggets"].apply(lambda x: x if len(x) <= max_nuggets else set([]))


    from cuttsum.pipeline import DedupedArticlesResource
    ded = DedupedArticlesResource()
    stats_df = ded.get_stats_df(event, corpus, extractor, thresh)
    stats_df["stream ids"] = stats_df["stream ids"].apply(lambda x: set(eval(x)))
    sid2match = {}
    for _, row in stats_df.iterrows():
        for sid in row["stream ids"]:
            sid2match[sid] = row["match"]

    all_ts = []
    all_docs = []
    new_docs = []
    for (sid, ts), doc in df.groupby(["stream id", "timestamp"]):
        if truncate is True:
            doc = doc.iloc[0:5]
#            print sub_doc
        if len(all_ts) > 0:
            assert ts >= all_ts[-1]
        all_ts.append(ts)
        if sid2match[sid] is True:
            new_docs.append(doc)
        all_docs.append(doc)

        
    df = pd.concat(new_docs)    
    print len(all_docs), len(new_docs)
    return df


def ds(df, sample_size=100):
    I = np.arange(len(df))
    np.random.shuffle(I)
    I = I[:sample_size]
    I = np.sort(I)
    return df.iloc[I]

def predict(task, event_stream, n_iter):

 #   task._keep_scores = True        
    pred = task.predict(event_stream)
    pred = ["SELECT" if p == SELECT else "SKIP" for p in pred]
#    scores = task._scores

  #  for score, action in zip(scores, pred):
  #      sel = math.exp(- ( score["SELECT"])) 
  #      nex = math.exp(- score["NEXT"]) 
  #      Z = sel + nex
  #      p_sel = 1. / (1. + sel) 
  #      p_sel = sel / Z
 #       p_nex = 1. / (1. + nex)          #
 #       p_nex = nex / Z

#        print p_sel, p_nex,  action
#p_nex, action

    all_nuggets = set()
    for nuggets in event_stream[1]["nuggets"].tolist():
        all_nuggets.update(nuggets)

    loss = 0        
    y_int_y_hat = 0
    size_y = 0
    size_y_hat = 0

    summary = []
    nuggets = set()
    for action, (_, sent) in izip(pred, event_stream[1].iterrows()):
        gain = len(sent["nuggets"] - nuggets)
        if action == "SELECT": 
            summary.append({
                "event": event_stream[0].query_id,
                "iter": n_iter,
                "update id": sent["update id"],
                "timestamp": sent["timestamp"],
                "gain": gain, 
                "nuggets": ",".join(sent["nuggets"]), 
                "update text": sent["pretty text"]
            })
            nuggets.update(sent["nuggets"])
        if gain > 0:
            oracle = "SELECT"
        else:
            oracle = "SKIP"

        if action == "SELECT" and oracle == "SELECT":
            y_int_y_hat += 1
            size_y += 1
            size_y_hat += 1
        elif action == "SELECT" and oracle == "SKIP":
            size_y_hat += 1
        elif action == "SKIP" and oracle == "SELECT":
            size_y += 1

    if size_y + size_y_hat == 0:
        loss = 1
    else:
        loss = 1 - float(y_int_y_hat) / (size_y + size_y_hat)
    

    if len(nuggets) > 0:
        egain = len(nuggets) / sum([1.0 if a == "SELECT" else 0.0 for a in pred])
    else:
        egain = 0        
    comp = len(nuggets) / float(len(all_nuggets)) 
    f1 = 2 * ( egain * comp) / (egain + comp) if egain + comp > 0 else 0

        
    return egain, comp, f1, loss, summary 

def do_work(train_instances, dev_instances, test_instances, sample_size, samples_per_event,
        gold_probs, iters, l2, log_time, semsims, dfdeltas,
        use_best_feats, use_i_only, use_abs_df, doc_condition, output_dir):
    

    vw = pyvw.vw(
        ("-l .5 --l2 {} --search 2 --search_task hook --ring_size 1024 " + \
         "--search_no_caching --noconstant --quiet").format(l2)) 
    task = vw.init_search_task(Summarizer)
    task.use_best_feats = use_best_feats
    task.use_i_only = use_i_only
    task.use_abs_df = use_abs_df
    task._doc_condition = doc_condition
    print "use best?", task.use_best_feats 
    print "use i only?", task.use_i_only
    print "use abs df?", task.use_abs_df
    print "use doc condition?", task._doc_condition



    all_scores = []
    all_weights = []


    for n_iter in xrange(1, iters + 1):
        task.total_loss = 0    

        random.shuffle(train_instances)
        print "iter", n_iter
        task.learn(train_instances)
        for i, inst in enumerate(dev_instances):
            egain, comp, f1, loss, _ = predict(task, inst, n_iter)
            print egain, comp, f1, loss
            all_scores.append(
                {"iter": n_iter, "E[gain]": egain, "Comp.": comp, "F1": f1, 
                 "Loss": loss}
            )
        df = pd.DataFrame(all_scores)
        df_u = df.groupby("iter").mean().reset_index(drop=True)
        print df_u    

        select_df, next_df = task.get_feature_weights()

        select_df["class"] = "SELECT"
        select_df["iter"] = n_iter

        next_df["class"] = "NEXT"
        next_df["iter"] = n_iter
        all_weights.append(select_df)
        all_weights.append(next_df)



    best_f1_iter = df_u["F1"].argmax() + 1
    best_egain_iter = df_u["E[gain]"].argmax() + 1
    best_comp_iter = df_u["Comp."].argmax() + 1
    best_loss_iter = df_u["Loss"].argmin() + 1

    weights_df = pd.concat(all_weights)


    all_summaries = []
#    all_scores = []

    F1_weights = weights_df[weights_df["iter"] == best_f1_iter]
    loss_weights = weights_df[weights_df["iter"] == best_loss_iter]
    egain_weights = weights_df[weights_df["iter"] == best_egain_iter]
    comp_weights = weights_df[weights_df["iter"] == best_comp_iter]

    def get_summaries(weights, run):
        print "Best", run
        task.set_weights(weights)
        for test_instance in test_instances:
            event = test_instance[0]
            df = test_instance[1]
            print event
            task._keep_scores = True
            task._scores = []
            predictions = task.predict(test_instance)
            assert len(predictions) == len(task._scores)

            for action, (_, row), ascore in zip(predictions, df.iterrows(), task._scores):
                if action == SELECT:
                  #  assert ascore["SELECT"] <= ascore["NEXT"]
                    print "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        event.query_num, "CUNLP", run, 
                        "-".join(row["update id"].split("-")[0:2]), 
                        row["update id"].split("-")[2], 
                        row["timestamp"], ascore)
                    all_summaries.append(
                        {"event": event.query_num, 
                         "team": "CUNLP",
                         "run": run,
                         "stream id": "-".join(row["update id"].split("-")[0:2]),
                         "sentence id": row["update id"].split("-")[2], 
                         "timestamp": row["timestamp"],
                         "confidence": row["probs"],
                         "partial": ascore,
                         "text": row["sent text"],
                         "pretty text": row["pretty text"]
                        })

    get_summaries(F1_weights, "L2S.F1")
    get_summaries(loss_weights, "L2S.Loss")
    get_summaries(egain_weights, "L2S.E[gain]")
    get_summaries(comp_weights, "L2S.Comp.")

            
    df = pd.DataFrame(all_summaries, 
        columns=["event", "team", "run", "stream id", "sentence id", 
                 "timestamp", "confidence", "partial", "pretty text", "text"])
    submission_path = os.path.join(output_dir, "submission.tsv")
    summary_path = os.path.join(output_dir, "summaries.tsv")
    f1_weights_path = os.path.join(output_dir, "weights.f1.tsv")
    loss_weights_path = os.path.join(output_dir, "weights.loss.tsv")
    egain_weights_path = os.path.join(output_dir, "weights.egain.tsv")
    comp_weights_path = os.path.join(output_dir, "weights.comp.tsv")

    scores_path = os.path.join(output_dir, "scores.tsv")

    no_text = ["event", "team", "run", "stream id", "sentence id", 
               "timestamp", "confidence", "partial"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df["confidence"] = df["confidence"].apply(lambda x: max(x, 0))
    with open(submission_path, "w") as f:
        df[no_text].to_csv(f, index=False, header=False, sep="\t")
    with open(summary_path, "w") as f:
        df.to_csv(f, index=False, sep="\t")

    with open(f1_weights_path, "w") as f:
        F1_weights.to_csv(f, index=False, sep="\t")
    with open(loss_weights_path, "w") as f:
        loss_weights.to_csv(f, index=False, sep="\t")
    with open(egain_weights_path, "w") as f:
        egain_weights.to_csv(f, index=False, sep="\t")
    with open(comp_weights_path, "w") as f:
        comp_weights.to_csv(f, index=False, sep="\t")
    with open(scores_path, "w") as f:
        df_u.to_csv(f, sep="\t", index=False)





    

def make_subsamples(streams, samples_per_stream, sample_size, dfdeltas, semsims):
    instances = []
    for sample in xrange(samples_per_stream):
        for event, stream in streams:
            while 1:
                sample_stream = ds(stream, sample_size=sample_size)
                if (sample_stream["nuggets"].apply(len) > 0).any():
                    break
            X_l = semsims[event.type].transform(
                sample_stream["stems"].apply(lambda x: ' '.join(x)).tolist())  
            instances.append((event, sample_stream, X_l, dfdeltas(event)))
    return instances



def main(test_event, sample_size, samples_per_event, gold_probs, iters, l2, 
    log_time, use_best_feats, use_i_only, use_abs_df, output_dir, 
    use_2015F, truncate, doc_condition):
    semsims = get_all_semsim()
    dfdeltas = get_dfdeltas()

    dev_ids = [19, 23, 27, 34, 35]

    streams = []
    for event in cuttsum.events.get_events():
        
        if event.query_num in set([24, 7, test_event] + dev_ids): continue
                #or event.query_num > 25: continue
        df = get_input_stream(
            event, gold_probs, use_2015F=use_2015F, truncate=truncate)
        streams.append((event, df))

    dev_instances = []
    for event in cuttsum.events.get_events():
        if event.query_num not in dev_ids: continue
        df = get_input_stream(
            event, gold_probs, use_2015F=use_2015F, truncate=truncate)

        X_l = semsims[event.type].transform(
            df["stems"].apply(lambda x: ' '.join(x)).tolist())  

        dev_instances.append((event, df, X_l, dfdeltas(event)))
        
    test_streams = []
    for event in cuttsum.events.get_events():
        if event.query_num != test_event: continue

        df = get_input_stream(
            event, gold_probs, use_2015F=use_2015F, truncate=truncate)

        X_l = semsims[event.type].transform(
            df["stems"].apply(lambda x: ' '.join(x)).tolist())  

        test_streams.append((event, df, X_l, dfdeltas(event)))
 

    train_instances = make_subsamples(streams, samples_per_event, 
        sample_size, dfdeltas, semsims)

    #dev_instances = make_subsamples(streams, samples_per_event, 
    #    sample_size, dfdeltas, semsims)


    job_results = do_work(
        train_instances, dev_instances, test_streams, 
        sample_size, samples_per_event, gold_probs, iters, l2, log_time, 
        semsims, dfdeltas, use_best_feats, use_i_only, use_abs_df, 
        doc_condition, output_dir)

    
if __name__ == u"__main__":

    import argparse

    parser = argparse.ArgumentParser()
#    parser.add_argument(u"--event-ids", type=int, nargs=u"+",
#                        help=u"event ids to select.")
    parser.add_argument(u"--sample-size", type=int,
                        default=100, 
                        help=u"downsample size for each training instance.") 
    parser.add_argument(
        u"--samples-per-event", type=int,default=10, 
        help=u"number of training instances to make from each event.")

    parser.add_argument(
        u"--gold-probs", type=bool, default=False, 
        help=u"Use gold nugget probability feature.")

    parser.add_argument(u"--iters", type=int,
                        default=10, 
                        help=u"Training iters") 

    parser.add_argument(u"--output-dir", type=str,
                        required=True, help="directory to write results.")
    parser.add_argument(u"--l2", type=float,
                        default=0, help="l2 weight")
    parser.add_argument(
        u"--best-feats", action="store_true", default=False, 
        help=u"Use best features")

    parser.add_argument(
        u"--i-only", action="store_true", default=False, 
        help=u"Use interactions only")
    parser.add_argument(
        u"--abs-df", action="store_true", default=False, 
        help=u"Use absolute value of df deltas.")


    parser.add_argument(
        u"--filter", action="store_true", default=False, 
        help=u"Use 2015F corpus.")

    parser.add_argument(
        u"--truncate", action="store_true", default=False, 
        help=u"Use first 5 sentences per doc.")

    parser.add_argument(
        u"--doc-condition", action="store_true", default=False, 
        help=u"Condition on number of selects in current document")
    
    parser.add_argument(
        u"--log-time", action="store_true", default=False, 
        help=u"Use log(t) feature")

    parser.add_argument(
        u"--test-event", type=int, required=True)

    args = parser.parse_args()

    main(args.test_event, args.sample_size, args.samples_per_event, 
         args.gold_probs, args.iters, args.l2, args.log_time,
         args.best_feats, args.i_only, args.abs_df, args.output_dir,
         args.filter, args.truncate, args.doc_condition)


