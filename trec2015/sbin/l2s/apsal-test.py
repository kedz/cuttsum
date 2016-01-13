import cuttsum.events
import cuttsum.corpora 
from cuttsum.pipeline import InputStreamResource
from cuttsum.classifiers import NuggetRegressor
import cuttsum.judgements
import pandas as pd
import numpy as np
from datetime import datetime
from cuttsum.misc import event2semsim
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
import os


def epoch(dt):
    return int((dt - datetime(1970, 1, 1)).total_seconds())

matches_df = cuttsum.judgements.get_merged_dataframe()

def get_input_stream(event, gold_probs, extractor="goose", thresh=.8, delay=None, topk=20):
    max_nuggets = 3
    corpus = cuttsum.corpora.get_raw_corpus(event)
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


def main(output_dir, sim_threshold, bucket_size, pref_offset): 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    dev_qids = set([19, 23, 27, 34, 35] + [7, 24])

    summary_data = []

    K_data = []
    for event in cuttsum.events.get_events():
        if event.query_num in dev_qids: continue
        

        print event

        semsim = event2semsim(event)
        istream = get_input_stream(event, False, extractor="goose", 
            thresh=.8, delay=None, topk=20)
        prev_time = 0
        cache = None

        clusters = []

        max_h = len(event.list_event_hours()) - 1


        for h, hour in enumerate(event.list_event_hours()):
            if h % bucket_size != 0 and h != max_h:
                continue
            
            current_time = epoch(hour)
            input_sents = istream[
                (istream["timestamp"] < current_time) & \
                (istream["timestamp"] >= prev_time)]
            len_select = input_sents["lemmas stopped"].apply(len) > 10
            input_sents = input_sents[len_select]

            if len(input_sents) <= 1: continue

            stems = input_sents["stems"].apply(lambda x: ' '.join(x)).tolist()
            X = semsim.transform(stems)
            probs = input_sents["probs"]
            p = probs.values
            K = -(1 - cosine_similarity(X))   
            K_ma = np.ma.masked_array(K, np.eye(K.shape[0]))
            Kmin = np.ma.min(K_ma)
            Kmax = np.ma.max(K_ma)
            median = np.ma.median(K_ma)[0]
            pref = np.minimum(p + median, -.05) 
            print "SYS TIME:", hour, "# SENTS:", K.shape[0], 
            print "min/median/max pref: {}/{}/{}".format(
                    pref.min(), np.median(pref), pref.max())

            #K_data.append({"min": Kmin, "max": Kmax, "median": median})
            K_data.append({"min": (pref).min(), "max": (pref).max(), 
                           "median": np.median((pref))})

            #print K
    #        continue
    #
            ap = AffinityPropagation(preference=pref-pref_offset, affinity="precomputed",
                    verbose=True, max_iter=50000)
            ap.fit(K)
    #        ##print input_sents["pretty text"]
    #       
            labels = ap.labels_ 
            if ap.cluster_centers_indices_ != None:
                for c in ap.cluster_centers_indices_:
                    if cache == None:
                        cache = X[c]
                        updates_df = input_sents.reset_index(
                                drop=True).iloc[c]
                        updates_df["query id"] = event.query_num
                        updates_df["system timestamp"] = current_time
                        summary_data.append(
                            updates_df[
                                ["query id", "stream id", "sent id", 
                                 "system timestamp", "sent text"]
                            ].to_frame().T)

                    else:
                        Ksum = cosine_similarity(cache, X[c])
                        
                        #print "MAX SIM", Ksum.max()
                        #print input_sents.reset_index(drop=True).iloc[c]["sent text"]

                        if Ksum.max() < sim_threshold:

                            cache = np.vstack([cache, X[c]])
                            updates_df = input_sents.reset_index(
                                    drop=True).iloc[c]
                            updates_df["query id"] = event.query_num
                            updates_df["system timestamp"] = current_time
                            summary_data.append(
                                updates_df[
                                    ["query id", "stream id", "sent id", 
                                     "system timestamp", "sent text"]
                                ].to_frame().T)

    #
    #        for l, i in enumerate(af.cluster_centers_indices_):
    #            support = np.sum(labels == l)
    #            center = input_sents.iloc[i][["update id", "sent text", "pretty text", "stems", "nuggets"]]
    #            center = center.to_dict()
    #            center["support"] = support
    #            center["timestamp"] = current_time
    #            clusters.append(center)
    #            


            prev_time = current_time
    #    df = pd.DataFrame(clusters, columns=["update id", "timestamp", "support", "sent text", "pretty text", "stems", "nuggets"])
    #
    #    import os
    #    dirname = "clusters"
    #    if not os.path.exists(dirname):
    #        os.makedirs(dirname)
    #
    #    with open(os.path.join(dirname, "{}.tsv".format(event.query_id)), "w") as f:
    #        df.to_csv(f, sep="\t", index=False)
    #
    df = pd.DataFrame(K_data, columns=["min", "max", "median"])
    print df
    print df.mean()
    print df.std()
    print df.max()
    df =  pd.concat(summary_data)
    df["conf"] = .5
    df["team id"] = "APSAL"
    df["run id"] = "sim{}_bs{}_off{}".format(
        sim_threshold, bucket_size, pref_offset)
    print df
    of = os.path.join(output_dir, "apsal" + "sim{}_bs{}_off{}.tsv".format(
                sim_threshold, bucket_size, pref_offset))
    cols = ["query id", "team id", "run id", "stream id", "sent id", 
            "system timestamp", "conf"]
    df[cols].to_csv(of, sep="\t", header=False, index=False) 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(u"--output-dir", type=str,
                        required=True, help="directory to write results.")
    parser.add_argument(
        u"--sim-cutoff", type=float, required=True)
    parser.add_argument(
        u"--bucket-size", type=float, required=True)
    parser.add_argument(
        u"--pref-offset", type=float, required=True)

    args = parser.parse_args()

    main(args.output_dir, args.sim_cutoff, 
            args.bucket_size, args.pref_offset)
