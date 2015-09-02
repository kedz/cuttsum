import pandas as pd
import sys
from datetime import datetime
import cuttsum.events
import os

events = {e.query_id: e for e in cuttsum.events.get_events()}

path = sys.argv[1]


with open(os.path.join(path, "summary.tsv"), "r") as f:
    df = pd.read_csv(f, sep="\t")

data = []
for name, group in df.groupby(["event", "iter"]):
    ts = group["update id"].apply(lambda x: x.split("-")[0]).tolist()
    
    sum_start = datetime.utcfromtimestamp(int(ts[0]))
    sum_end = datetime.utcfromtimestamp(int(ts[-1]))
    event_start = events[name[0]].start
    event_end = events[name[0]].end
    sum_dur = (sum_end - sum_start).total_seconds()
    event_dur = (event_end - event_start).total_seconds()
    burnout =  (event_end - sum_end).total_seconds() / (60. * 60.)
    warmup =  (sum_start - event_start).total_seconds() / (60. * 60.)
    data.append({"event": name[0], "iter": name[1], 
        "time coverage": sum_dur / float(event_dur), "burnout hrs": burnout, "warmup hrs": warmup, "length": len(group)})
    print name, warmup
stats_df = pd.DataFrame(data)[["iter", "time coverage", "burnout hrs", "warmup hrs", "length"]].groupby("iter").mean()

def fmeasure(p, r):
    return 2 * (p * r) / (p + r)
print stats_df
exit()

with open(os.path.join(path, "scores.tsv"), "r") as f:
    df = pd.read_csv(f, sep="\t")

    #for event, panel in df.groupby("event"):
    #    print event
    #    print panel




    mean_df = pd.concat([group.mean().to_frame().transpose()
                         for niter, group in df.groupby("iter")])
    mean_df["F1"] = fmeasure(mean_df["E[gain]"].values, mean_df["Comp."])
df = pd.concat([stats_df, mean_df.set_index("iter")[["E[gain]", "Comp.", "Loss", "F1", "Avg. Train F1", "Avg. Train Loss"]]], axis=1, join='inner')

print df
