import cuttsum.events
import cuttsum.corpora
from cuttsum.pipeline import ArticlesResource
import numpy as np
from scipy import linalg

event = cuttsum.events.get_events()[0]
corpus = cuttsum.corpora.EnglishAndUnknown2013()
extractor = "gold" 
res = ArticlesResource()

from sklearn.feature_extraction import DictVectorizer


def extract_sentence_features_gold(df, vec):
    all_nuggets = set(
        n for ns in df["nuggets"].tolist() for n in ns)
    X_dict = df["nuggets"].apply(
        lambda x: {key: 1 if key in x else -1 for key in all_nuggets}).tolist()
    #vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(X_dict)
    return X, vec
    
w = None
k = 5
m = 10
cache = {}

found_nuggets = set()

possible_nuggets = set(['VMTS13.01.064', 'VMTS13.01.058', 'VMTS13.01.056', 'VMTS13.01.055', 'VMTS13.01.054', 'VMTS13.01.052', 'VMTS13.01.051', 'VMTS13.01.050', 'VMTS13.01.062', 'VMTS13.01.088', 'VMTS13.01.077', 'VMTS13.01.065', 'VMTS13.01.078', 'VMTS13.01.086', 'VMTS13.01.069', 'VMTS13.01.099' "VMTS13.01.060", 'VMTS13.01.092', 'VMTS13.01.091', 'VMTS13.01.090', 'VMTS13.01.097', 'VMTS13.01.059', 'VMTS13.01.099', 'VMTS13.01.071', 'VMTS13.01.070', 'VMTS13.01.073', 'VMTS13.01.061', 'VMTS13.01.075', 'VMTS13.01.067', 'VMTS13.01.076', 'VMTS13.01.104', 'VMTS13.01.068', 'VMTS13.01.106', 'VMTS13.01.103', 'VMTS13.01.063', 'VMTS13.01.060', 'VMTS13.01.084', 'VMTS13.01.085', 'VMTS13.01.087', 'VMTS13.01.081', 'VMTS13.01.074'])

print "There are {} nuggets.".format(len(possible_nuggets))


all_matches = cuttsum.judgements.get_merged_dataframe()
matches = all_matches[all_matches["query id"] == event.query_id]
nugget_ids = set([n for n in matches["nugget id"].tolist()])
print nugget_ids
vec = DictVectorizer(sparse=False)
n_feats = 4
vec.fit_transform({key: 1 for key in possible_nuggets})
w = np.random.uniform(size=(n_feats))

samples = []

summary_rows = []

se_overtime = []

alpha = .2

add_me = set()

for df in res.dataframe_iter(event, corpus, extractor, include_matches="soft"):
    add_me.update(set(n for ns in df["nuggets"].apply(lambda x: x.difference(possible_nuggets)).tolist() for n in ns))
    df["nuggets"] = df["nuggets"].apply(lambda x: x.intersection(possible_nuggets))
    df = df[df["nuggets"].apply(len) > 0]
    df.reset_index(drop=True, inplace=True)
    if len(df) == 0:
        continue
    print df[["sent text", "nuggets"]]
    
    X_dict = df["nuggets"].apply(
        lambda x: {key: 1 for key in x}).tolist()

    X_sent = vec.transform(X_dict)

    for i in xrange(m):

        X_cache = vec.transform(cache)
        X = np.concatenate([
            len(summary_rows) * np.ones((X_sent.shape[0],))[:, None],
            np.logical_and(X_sent==1, X_cache == 1).astype("int32").sum(axis=1)[:,None],
            X_sent.sum(axis=1)[:, None],
            np.ones((X_sent.shape[0],))[:, None]], axis=1)
        print "X"
        print X
        #print X_sent.shape
        #print X_cache.shape
        #print X.shape

        oracle_gain = df["nuggets"].apply(lambda x: len(x.union(found_nuggets)) - len(found_nuggets)).values
        aug_oracle_gain = np.concatenate([oracle_gain, [alpha]])
        #print aug_oracle_gain
        #exit()
        #print oracle_gain

        policy_scores = np.dot(X, w)
        policy_argmin = np.argmin(policy_scores)
        policy_min = policy_scores[policy_argmin]
        print "n(S)", found_nuggets
        print "|S|", len(summary_rows)
        print "F(S) =", len(cache)
        print "POLICY ARGMIN", policy_argmin
        print "POLICY MIN", policy_min
        policy_gain = oracle_gain[policy_argmin]
        print "POLICY GAIN", policy_gain
        max_oracle_gain = aug_oracle_gain.max()
        print "ORACLE GAIN", max_oracle_gain
        print "se", 
        se = (max_oracle_gain - policy_gain - policy_min)**2
        print se
        #print found_nuggets
        policy_gain = policy_gain if policy_min <= 0 else alpha

        correct = 1 if policy_gain > 0 or (policy_min >= 0 and max_oracle_gain == 0) else 0
        se_overtime.append({
            "se": se, 
            "oracle gain": max_oracle_gain, 
            "policy gain": policy_gain, 
            "delta gain": max_oracle_gain - oracle_gain[policy_argmin], 
            "correct": correct})

        costs = np.array([max_oracle_gain - gain for gain in oracle_gain])
        
        


        print "COST", costs
        weights = np.array([(1 - 1./k)**(m-i) for _ in costs])
        #weights = np.array([1. for _ in costs])
        if policy_min < 0:
            print "Appending", df.loc[policy_argmin, "sent text"].encode("utf-8")
            summary_rows.append(df.loc[policy_argmin])
            for nid in df.loc[policy_argmin, "nuggets"]:
                cache[nid] = 1
                found_nuggets.add(nid)
        samples.append((X, costs, weights))
        

        I_new = range(policy_argmin) + range(policy_argmin + 1, len(df))
        df = df.loc[I_new]
        df.reset_index(inplace=True, drop=True)
        
        X_sent = X_sent[I_new]

        if len(summary_rows) > m or len(cache) == len(possible_nuggets):

            print "RESET"
            summary_rows = []
            cache = {}
            found_nuggets = set()

        if len(df) == 0:
            break    
        if max_oracle_gain == 0:
            break
        if policy_min >= 0:
            break
        #print df["nuggets"]
        #print X_cache
        
    X_sample = np.concatenate([sam[0] for sam in samples], axis=0)
    print X_sample.shape
    C_sample = np.concatenate([sam[1] for sam in samples], axis=0)
    W_sample = np.concatenate([sam[2] for sam in samples], axis=0)
    W_sqr = np.sqrt(W_sample)
    Xw = X_sample * W_sqr[:, None]
    cw = C_sample * W_sqr

    w_new = linalg.lstsq(Xw, cw)[0]

    #print w_new
    #print ((w_new - w)**2).sum()
    #print cache
    w = w_new
    print w



#    for i in xrange(k):
        
    
#    print df

import pandas as pd

cost_df = pd.DataFrame([{"cost": i } for i in C_sample])

with open("se.csv", "w") as f:
    pd.DataFrame(se_overtime).to_csv(f, index=False)

with open("costs.csv", "w") as f:
    cost_df.to_csv(f, index=False)

print w[0:44]
print 
print w[44:88]
print
print w[88:]

print "ADD ME"
print add_me
