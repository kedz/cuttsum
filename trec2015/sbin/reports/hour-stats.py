import cuttsum.events
import cuttsum.corpora
from cuttsum.trecdata import SCChunkResource
from cuttsum.pipeline import ArticlesResource, DedupedArticlesResource
import os
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pylab as plt
plt.style.use('ggplot')
pd.set_option('display.width', 200)
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')

def unix_time(dt):
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return int(delta.total_seconds())


articles_res = ArticlesResource()
#ded_articles_res = DedupedArticlesResource()
data = []
events = []
for event in cuttsum.events.get_events():
    if event.query_num == 7: continue
    if event.query_num > 25: continue
    events.append(event)
    corpus = cuttsum.corpora.get_raw_corpus(event)
    hours = event.list_event_hours()

    hour2goose = defaultdict(int)

    

    for hour in hours:
        path = articles_res.get_chunk_path(event, "goose", hour)
        if path is None:
            continue
        fname = os.path.split(path)[1]
        num_goose = int(fname.split("-")[0])
        hour2goose[hour] = num_goose
        #print event.fs_name(), hour.hour, num_goose
        #if hour.hour == 0:
        #    old_hour = 23
        #else:
        #    old_hour = hour.hour - 1
        prev = hour2goose.get(hour - timedelta(hours=1), 0)
        data.append({"hour": hour.hour, "event": event.query_id, "df": (num_goose-prev) / max(1, prev) })

agg = []
df = pd.DataFrame(data)
import numpy as np

stddf_data = []

for event in cuttsum.events.get_events():
    if event.query_num == 7: continue
    print event
    df_u = df[df["event"] != event.query_id].groupby("hour").mean()
    df_std = df[df["event"] != event.query_id].groupby("hour").std()
    print df_u
    print df_u.iloc[3]
    corpus = cuttsum.corpora.get_raw_corpus(event)
    hours = event.list_event_hours()
    for hour in hours:
        path = articles_res.get_chunk_path(event, "goose", hour)
        if path is None:
            continue
        fname = os.path.split(path)[1]
        num_goose = int(fname.split("-")[0])
        hour2goose[hour] = num_goose
        #print event.fs_name(), hour.hour, num_goose
        #if hour.hour == 0:
        #    old_hour = 23
        #else:
        #    old_hour = hour.hour - 1
        print event.query_id, hour, 
        prev = hour2goose.get(hour - timedelta(hours=1), 0)
        time_we_can_know_this = hour + timedelta(hours=1)
        print unix_time(time_we_can_know_this),
        print (num_goose-prev) / max(1., prev), ((num_goose-prev) / max(1., prev) - df_u["df"].iloc[hour.hour]) / df_std["df"].iloc[hour.hour]
        stddf = ((num_goose-prev) / max(1., prev) - df_u["df"].iloc[hour.hour]) / df_std["df"].iloc[hour.hour]

        stddf_data.append({"event": event.query_id, "hour": unix_time(time_we_can_know_this), "df delta": stddf})
        #data.append({"hour": hour.hour, "event": event.query_id, "df": (num_goose-prev) / max(1, prev) })

with open("doc_freqs.tsv", "w") as f:
    df = pd.DataFrame(stddf_data, columns=["event", "hour", "df delta"])
    df.to_csv(f, sep="\t", index=False)
exit()


df_u = df.groupby(["hour"]).mean()
agg=[df_u.values]
cols = ["mean"]
for event, e_df in df.groupby("event"):
    e_u = e_df.groupby("hour").mean() 
    print e_u
    e_u_d = e_u - df_u
    agg.append(e_u.values)
    agg.append(e_u_d.values)
    cols.append(event.query_id)
    cols.append(event.query_id + " (del.)")
print pd.DataFrame(np.hstack(agg), columns=cols)
