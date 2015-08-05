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


class Summarizer(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw , sch, num_actions)
        #sch.set_options( sch.AUTO_HAMMING_LOSS )
        self._with_scores = False
        self._loss = 1
    
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

  #for i, feat in enumerate(self.basic_cols()):
            #fw.append(("b:" + feat, w))
        #fw.append(("n:bias", self.vw.get_weight(ex.feature("n", 0))))


    def set_loss(self, loss):
        self._loss = loss

    def make_example(self, sent, cache, cache_in, days):
        df = sent.to_frame().transpose()
        feats = {
            "b": [(k, df[k].tolist()[0]) for k in basic_cols],
            "c": [(k, df[k].tolist()[0]) for k in sum_cols],
            "q": [(k, df[k].tolist()[0]) for k in query_fw_cols],
            "l": [(k, df[k].tolist()[0]) for k in lm_cols],
            "s": [(k, df[k].tolist()[0]) for k in stream_cols],
            "p": [("probs",sent["probs"])],
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

        feats["I"] = []
        for ns in ["b", "c", "q", "l", "s", "g"]:
            for feat, val in feats[ns]:
                feats["I"].append(("{}^time^prob".format(feat), val * sent["probs"] * days))
                feats["I"].append(("{}^prob".format(feat), val * sent["probs"]))
                feats["I"].append(("{}^time".format(feat), val * days))
 
        ifeats = {'a': []}
        for ns in ["b", "c", "q", "l", "s", "g", "p", "I"]:
            for key, val in feats[ns]: 
                ifeats['a'].append((fmap[key], val))
        ifeats['a'].append((fmap["CONSTANT"], 1))
        return self.example(ifeats)

    def _run(self, (event, df_stream)):

        nuggets = set()
        cache = None
        cache_in = None 
        output = []

        n = 0
        loss = 0

        for _, sent in df_stream.iterrows():
            days = (datetime.utcfromtimestamp(int(sent["timestamp"])) - event.start).total_seconds() / (60. * 60. * 24.)
            n += 1
            gain = len(sent["nuggets"] - nuggets)
            if self.sch.predict_needs_example():
                examples = self.make_example(sent, cache, cache_in, days)
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
            if pred != oracle:
                if oracle == SELECT:
                    loss += self._loss
                else:
                    loss += 1
            #if self._with_scores is True:
            #    print "examining:", sent["pretty text"]        

            if pred == SELECT:
                nuggets.update(sent["nuggets"])
                if cache is None:
                    cache = sent.to_frame().transpose()
                else:
                    cache = pd.concat([cache, sent.to_frame().transpose()])
#            else:
#                if cache_in is None:
#                    cache_in = sent.to_frame().transpose()
#                else:
#                    cache_in = pd.concat([cache_in, sent.to_frame().transpose()])

        self.sch.loss(loss)
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
                    

def start_worker(sample_size, samples_per_event, gold_probs, iters, l2):
    rank = MPI.COMM_WORLD.Get_rank()
    status = MPI.Status() # get MPI status object
    job_results = None
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
                sample_size, samples_per_event, gold_probs, iters, l2)

        if tag == tags.WORKER_STOP:
            break
    
 
def do_work(training_events, test_event, sample_size, samples_per_event,
        gold_probs, iters, l2):
    
    training_streams = []
    summary = []

    for event in training_events:
        df = get_input_stream(event, gold_probs)
        training_streams.append((event, df))

    test_stream = (test_event, get_input_stream(test_event, gold_probs)) 

    vw = pyvw.vw(
        ("--l2 {} --search 2 --search_task hook --ring_size 1024 " + \
         "--search_no_caching --noconstant --quiet").format(l2)) 
    
    task = vw.init_search_task(Summarizer)
    all_scores = []
    all_weights = []
    for n_iter in xrange(1, iters + 1):
    
        instances = [(event, ds(stream, sample_size=sample_size))
                     for event, stream in training_streams
                     for sample in xrange(samples_per_event)]
        random.shuffle(instances)
        for i, inst in enumerate(instances):
            print "{}.{}.{}/{}".format(
                test_event.fs_name(), n_iter, i, len(instances))
            task.learn([inst])
        print "{}.{}.p".format(
            test_event.fs_name(), n_iter)
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
        nuggets = set()
        for action, (_, sent) in izip(pred, test_stream[1].iterrows()):
            if action != "SELECT": continue 
            gain = len(sent["nuggets"] - nuggets)
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

        if len(nuggets) > 0:
            egain = len(nuggets) / sum([1.0 if a == "SELECT" else 0.0 for a in pred])
        else:
            egain = 0        
        comp = len(nuggets) / float(len(all_nuggets)) 
        
        all_scores.append({"iter": n_iter, "Comp.": comp,
                           "E[gain]": egain, "Loss": loss})        

        print "{}.{}.p E[gain]={:0.6f} Comp.={:0.6f}".format(
            test_event.fs_name(), n_iter, egain, comp)

    scores_df = pd.DataFrame(all_scores, columns=["iter", "E[gain]", "Comp.", "Loss"])
    weights_df = pd.concat(all_weights)
    weights_df["event"] = test_event.query_id
    scores_df["event"] = test_event.query_id
    summary_df = pd.DataFrame(
        summary, 
        columns=["iter", "event", "update id", "timestamp", "gain", 
                 "update text", "nuggets"])
    return scores_df, weights_df, summary_df


def get_input_stream(event, gold_probs, extractor="goose", thresh=.8, delay=None, topk=20):
    corpus = cuttsum.corpora.get_raw_corpus(event)
    res = InputStreamResource()
    df = pd.concat(
        res.get_dataframes(event, corpus, extractor, thresh, delay, topk))
    df["true probs"] = df["nugget probs"].apply(lambda x: [val for key, val in x.items()] +[0])
    df["true probs"] = df["true probs"].apply(lambda x: np.max(x))
    df.loc[(df["n conf"] == 1) & (df["nuggets"].apply(len) == 0), "true probs"] = 0

    if gold_probs is True:
        df["probs"] = df["true probs"]
    else:
        df["probs"] = NuggetRegressor().predict(event, df)   
    
    df["nuggets"] = df["nugget probs"].apply(
        lambda x: set([key for key, val in x.items() if val > .5]))
    return df


def ds(df, sample_size=100):
    I = np.arange(len(df))
    np.random.shuffle(I)
    I = I[:sample_size]
    I = np.sort(I)
    return df.iloc[I]




if __name__ == u"__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(u"--event-ids", type=int, nargs=u"+",
                        help=u"event ids to select.")
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
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    if size == 1:
        print "Must be run with at least 2 processes!"
        exit()
    if rank == 0:
        start_manager(args.event_ids, args.output_dir)
    else:
        start_worker(args.sample_size, args.samples_per_event, 
                     args.gold_probs, args.iters, args.l2)
