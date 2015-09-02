import streamcorpus as sc
import cuttsum.events
import cuttsum.corpora
from cuttsum.trecdata import SCChunkResource
from cuttsum.pipeline import ArticlesResource, DedupedArticlesResource
import os
import pandas as pd
from datetime import datetime
from collections import defaultdict
import matplotlib.pylab as plt
plt.style.use('ggplot')
pd.set_option('display.max_rows', 500)

pd.set_option('display.width', 200)

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')


def format_int(x):
    return locale.format("%d", x, grouping=True)


def epoch(dt):
    return int((dt - datetime(1970, 1, 1)).total_seconds())

chunk_res = SCChunkResource()
articles_res = ArticlesResource()
ded_articles_res = DedupedArticlesResource()
data = []

event2ids = defaultdict(set)
fltr_event2ids = defaultdict(set)
for event in cuttsum.events.get_events():
    
    corpus = cuttsum.corpora.get_raw_corpus(event)
    hours = event.list_event_hours()

    hour2ded = defaultdict(int)
    hour2ded_fltr = defaultdict(int)
    ded_df = ded_articles_res.get_stats_df(event, corpus, "goose", .8)
    
    if ded_df is not None:  

        if event.query_num > 25:
            for ids in ded_df["stream ids"].apply(eval).tolist():
                for id1 in ids:
                    event2ids[event.fs_name()].add(id1)

        for _, row in ded_df.iterrows():
            dt = datetime.utcfromtimestamp(row["earliest"])
            hour = datetime(dt.year, dt.month, dt.day, dt.hour)
            hour2ded[hour] += 1
            if row["match"] == True:
                hour2ded_fltr[hour] += 1

    hour2goose = defaultdict(int)

    for hour in hours:
        path = articles_res.get_chunk_path(event, "goose", hour, corpus)
        if path is None:
            continue
        #print path
        fname = os.path.split(path)[1]
        num_goose = int(fname.split("-")[0])
        hour2goose[hour] = num_goose
#    goose_df = articles_res.get_stats_df(event, "goose")
#    if goose_df is not None:
#        for _, row in goose_df.iterrows():
#            dt = datetime.utcfromtimestamp(row["hour"])
#            hour = datetime(dt.year, dt.month, dt.day, dt.hour)
#            hour2goose[hour] = row["goose articles"]
                       
 
    for hour in hours:
        raw_chunks = chunk_res.get_chunks_for_hour(hour, corpus, event)
        num_raw_si = 0
        
        for chunk in raw_chunks:
            fname = os.path.split(chunk)[1]
            num_raw_si += int(fname.split("-")[1])
        #num_fltr_si = len(articles_res.get_si(event, corpus, "goose", hour))
        data.append({
            "event": event.query_id,
            "title": event.title,
            "hour": hour,
            "raw articles": num_raw_si,
            "goose articles": hour2goose[hour],
            "deduped articles": hour2ded[hour],
            "deduped match articles": hour2ded_fltr[hour],
        })


for event in cuttsum.events.get_events():
    if event.query_num < 26: continue    
    
    corpus = cuttsum.corpora.FilteredTS2015()
    hours = event.list_event_hours()

    hour2ded = defaultdict(int)
    hour2ded_fltr = defaultdict(int)
    ded_df = ded_articles_res.get_stats_df(event, corpus, "goose", .8)


    if ded_df is not None:  
        for ids in ded_df["stream ids"].apply(eval).tolist():
            for id1 in ids:
                fltr_event2ids[event.fs_name()].add(id1)



        for _, row in ded_df.iterrows():
            dt = datetime.utcfromtimestamp(row["earliest"])
            hour = datetime(dt.year, dt.month, dt.day, dt.hour)
            hour2ded[hour] += 1
            if row["match"] == True:
                hour2ded_fltr[hour] += 1

    hour2goose = defaultdict(int)

    for hour in hours:
        path = articles_res.get_chunk_path(event, "goose", hour, corpus)
        if path is None:
            continue
        print path
        fname = os.path.split(path)[1]
        num_goose = int(fname.split("-")[0])
        hour2goose[hour] = num_goose
#    goose_df = articles_res.get_stats_df(event, "goose")
#    if goose_df is not None:
#        for _, row in goose_df.iterrows():
#            dt = datetime.utcfromtimestamp(row["hour"])
#            hour = datetime(dt.year, dt.month, dt.day, dt.hour)
#            hour2goose[hour] = row["goose articles"]
                       
 
    for hour in hours:
        print hour
        raw_chunks = chunk_res.get_chunks_for_hour(hour, corpus, event)
        num_raw_si = 0
        
        for chunk in raw_chunks:
            fname = os.path.split(chunk)[1]
            #num_raw_si += int(fname.split("-")[1])
            with sc.Chunk(path=chunk, mode="rb", message=corpus.sc_msg()) as c:
                for si in c:
                    num_raw_si += 1
        #num_fltr_si = len(articles_res.get_si(event, corpus, "goose", hour))
        data.append({
            "event": event.query_id + " (filtered)",
            "title": event.title,
            "hour": hour,
            "raw articles": num_raw_si,
            "goose articles": hour2goose[hour],
            "deduped articles": hour2ded[hour],
            "deduped match articles": hour2ded_fltr[hour],
        })



df = pd.DataFrame(data)
cols = ["raw articles", "goose articles", "deduped articles", 
        "deduped match articles"]

df_sum = df.groupby("event")[cols].sum()


df_sum["raw articles"] = df_sum["raw articles"].apply(format_int)
df_sum["goose articles"] = df_sum["goose articles"].apply(format_int)
df_sum["deduped articles"] = df_sum["deduped articles"].apply(format_int)
df_sum["deduped match articles"] = df_sum["deduped match articles"].apply(format_int)


print df_sum
print


coverage = []
for event in cuttsum.events.get_events():
    if event.query_num < 26: continue 

    isect = event2ids[event.fs_name()].intersection(fltr_event2ids[event.fs_name()])
    n_isect = len(isect)
    n_unfltr = max(len(event2ids[event.fs_name()]), 1)
    n_fltr = max(len(fltr_event2ids[event.fs_name()]), 1)
    print event.fs_name()
    print n_isect, float(n_isect) / n_fltr, float(n_isect) / n_unfltr
    coverage.append({
        "event": event.query_id,
        "intersection": n_isect,
        "isect/n_2015F": float(n_isect) / n_fltr,
        "isect/n_2014": float(n_isect) / n_unfltr,
    })

df = pd.DataFrame(coverage)
df_u = df.mean()
df_u["event"] = "mean"
print pd.concat([df, df_u.to_frame().T]).set_index("event")

exit()
with open("article_count.tex", "w") as f:
    f.write(df_sum.to_latex())

import os
if not os.path.exists("plots"):
    os.makedirs("plots")

import cuttsum.judgements
ndf = cuttsum.judgements.get_merged_dataframe()
for (event, title), group in df.groupby(["event", "title"]):
    matches = ndf[ndf["query id"] == event]

    #fig = plt.figure()
    group = group.set_index(["hour"])
    #ax = group[["goose articles", "deduped articles", "deduped match articles"]].plot()
    linex = epoch(group.index[10])
    ax = plt.plot(group.index, group["goose articles"], label="goose")
    ax = plt.plot(group.index, group["deduped articles"], label="dedupe")
    ax = plt.plot(group.index, group["deduped match articles"], label="dedupe qmatch")
    for nugget, ngroup in matches.groupby("nugget id"):
        times = ngroup["update id"].apply(lambda x: datetime.utcfromtimestamp(int(x.split("-")[0])))
        #ngroup = ngroup.sort("timestamp")
        times.sort()
        times = times.reset_index(drop=True)
        if len(times) == 0: continue
        plt.plot_date(
            (times[0], times[0]), 
            (0, plt.ylim()[1]), 
            '--', color="black", linewidth=.5, alpha=.5)
    plt.gcf().autofmt_xdate()
    plt.gcf().suptitle(title)
    plt.gcf().savefig(os.path.join("plots", "{}-stream.png".format(event)))
    plt.close("all")


