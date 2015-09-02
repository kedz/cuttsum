import sys
import os
import pandas as pd
from collections import defaultdict
import numpy as np

dirname = sys.argv[1]

path = os.path.join(dirname, "weights.tsv")
with open(path ,"r") as f:
    df = pd.read_csv(f, sep="\t")
df = df[df["iter"] == 5]
fc2r = defaultdict(list)
features = set()


for event, event_df in df.groupby("event"):
    pos_df = event_df[event_df["class"] == "SELECT"]
    event_df.loc[event_df["class"] == "SELECT", "rank"] = pos_df["weight"].argsort()
    for _, row in event_df.loc[event_df["class"] == "SELECT"][["name", "weight", "rank"]].iterrows():
        clazz = "SELECT"
        feature = row["name"]
        rank = row["rank"]
        fc2r[(feature, clazz)].append(rank)
        features.add(feature)
    

    neg_df = event_df[event_df["class"] == "NEXT"]
    event_df.loc[event_df["class"] == "NEXT", "rank"] = neg_df["weight"].argsort()
    for _, row in event_df.loc[event_df["class"] == "NEXT"][["name", "weight", "rank"]].iterrows():
        clazz = "NEXT"
        feature = row["name"]
        rank = row["rank"]
        fc2r[(feature, clazz)].append(rank)
        features.add(feature)

f2d = {}
for feature in features:
    sel_u = np.mean(fc2r[(feature, "SELECT")])
    next_u = np.mean(fc2r[(feature, "NEXT")])
    diff = max(sel_u, next_u) - min(sel_u, next_u)
    f2d[feature] = diff    

print
feat_diff = sorted(f2d.items(), key=lambda x: x[1])
for feat, diff in feat_diff[-50:]:
    print feat
