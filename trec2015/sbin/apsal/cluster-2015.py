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

    return df


for event in cuttsum.events.get_events():
    if event.query_num < 26: continue
    print event
    semsim = event2semsim(event)
    istream = get_input_stream(event, False, extractor="goose", 
        thresh=.8, delay=None, topk=20)
    prev_time = 0
    cache = None

    clusters = []
    for hour in event.list_event_hours():
        current_time = epoch(hour)
        input_sents = istream[
            (istream["timestamp"] < current_time) & \
            (istream["timestamp"] >= prev_time)]
        input_sents = input_sents[input_sents["lemmas stopped"].apply(len) > 5]
        if len(input_sents) <= 1: continue

        stems = input_sents["stems"].apply(lambda x: ' '.join(x)).tolist()
        X = semsim.transform(stems)
        probs = input_sents["probs"]
        p = probs.values
        #p[p < 0] = 0
        #p[p > 1] = 1
        from scipy.special import expit
        from sklearn.metrics.pairwise import cosine_similarity
        p = expit(p)
        pref = p 
        K = -(1 - cosine_similarity(X))**2 - 20
        K_ma = np.ma.masked_array(K, np.eye(K.shape[0]))

        median = np.ma.median(K_ma)
        

        af = AffinityPropagation(preference=pref + median).fit(X)
        ##print input_sents["pretty text"]
       
        labels = af.labels_ 
        print len(input_sents), af.cluster_centers_indices_.shape

        for l, i in enumerate(af.cluster_centers_indices_):
            support = np.sum(labels == l)
            center = input_sents.iloc[i][["update id", "sent text", "pretty text", "stems", "nuggets", "probs"]]
            center = center.to_dict()
            center["support"] = support
            center["timestamp"] = current_time
            clusters.append(center)
            
#            if cache is None:
#                cache = X[i]

#                print i, np.sum(labels == l), input_sents.iloc[i]["pretty text"]
#            else:
#                sim = cosine_similarity(cache, X[i])
#                if sim[0] < threshold:
#                    print i, np.sum(labels == l), input_sents.iloc[i]["pretty text"]
#                    cache = np.vstack([cache, X[i]]) 


        prev_time = current_time
    df = pd.DataFrame(clusters, columns=["update id", "timestamp", "support", "sent text", "pretty text", "stems", "nuggets", "probs"])

    import os
    dirname = "clusters-2015"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, "{}.tsv".format(event.query_id)), "w") as f:
        df.to_csv(f, sep="\t", index=False)


