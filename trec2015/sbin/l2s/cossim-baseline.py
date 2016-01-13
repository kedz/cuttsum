import cuttsum.events
import cuttsum.corpora
from cuttsum.pipeline import InputStreamResource
import numpy as np
import pandas as pd
import random
from cuttsum.classifiers import NuggetRegressor
from cuttsum.misc import event2semsim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
import os
np.random.seed(42)

matches_df = cuttsum.judgements.get_merged_dataframe()

def get_input_stream(event, gold_probs, extractor="goose", thresh=.8, delay=None, topk=20, 
        use_2015F=False, truncate=1):
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
        #if truncate is True:
        doc = doc.iloc[0:truncate]
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


def main(output_dir, use_2015F, query_nums, sim_cutoff=.5, use_semsim=False):

    results = []
    for query_num in query_nums:
        event = [e for e in cuttsum.events.get_events() 
                 if e.query_num == query_num][0]
        print event

        gold_probs = False
        df = get_input_stream(
            event, gold_probs, use_2015F=use_2015F, truncate=1)
        df = df.loc[df["stems"].apply(len) >= 10]
        df = df.reset_index(drop=True)

        print df[["update id", "sent text"]]

        if use_semsim:
            semsims = get_all_semsim()
            X_l = semsims[event.type].transform(
                    df["stems"].apply(lambda x: ' '.join(x)).tolist())  
        
            K = cosine_similarity(X_l)
        else:
            Xtf = []
            for stems in df["stems"].tolist():
                sc = {}
                for stem in stems:
                    sc[stem] = sc.get(stem, 0) + 1
                Xtf.append(sc)
            dv = DictVectorizer()
            Xtf = dv.fit_transform(Xtf)
            print Xtf
            K = cosine_similarity(Xtf)

        print K
        S = [0]
        for s in range(1, len(df)):
            max_prev_sim = K[s,:s].max()
            if max_prev_sim < sim_cutoff:
                S.append(s)
        for sent_text in df.iloc[S]["pretty text"].tolist():
            print sent_text
        for _, row in df.iloc[S].iterrows():
            d = row.to_dict()
            d["query id"] = query_num
            d["conf"] = .5
            d["team"] = "CUNLP"
            d["run id"] = "{}.c{}".format(
                "sem" if use_semsim else "bow", sim_cutoff)
            results.append(d)
    df = pd.DataFrame(results, columns=["query id", "team", "run id",
        "stream id", "sent id", "timestamp", "conf"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    o = os.path.join(output_dir, "{}.c{}.tsv".format(
        "sem" if use_semsim else "bow", sim_cutoff))
    df.to_csv(o, sep="\t", header=False, index=False)

##1   CUNLP   L2S.F1  1330002360-e6b4f40b0cd290d8290d3b4d76e3c6ad 5   1330002360  1.18571607163   0.0153079582378 
#1   CUNLP   L2S.F1  1330041480-dc14aa4b986c960cf5ac0ff980d483dd 1   1330041480  0.748650560205  0.0164099317044

if __name__ == u"__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(u"--output-dir", type=str,
                        required=True, help="directory to write results.")
    parser.add_argument(
        u"--filter", action="store_true", default=False, 
        help=u"Use 2015F corpus.")
    parser.add_argument(
            u"--test-event", type=int, nargs="+", required=True)
    parser.add_argument(
        u"--sim-cutoff", type=float, required=True)
    parser.add_argument(
        u"--sem-sim", action="store_true", default=False)

    args = parser.parse_args()
    print args.test_event
    main(args.output_dir, args.filter, args.test_event, 
         sim_cutoff=args.sim_cutoff, use_semsim=args.sem_sim)
