import streamcorpus as sc
import cuttsum.judgements
import cuttsum.events
import cuttsum.corpora
from cuttsum.misc import stringify_streamcorpus_sentence
import pandas as pd

event = cuttsum.events.get_events(by_query_ids=["TS13.1"])[0]
corpus = cuttsum.corpora.EnglishAndUnknown2013()

paths = ["/home/kedz/projects2015/trec2015/data/articles/gold/2012_buenos_aires_rail_disaster/2012-02-22-12.sc.gz",
         "/home/kedz/projects2015/trec2015/data/articles/gold/2012_buenos_aires_rail_disaster/2012-02-22-13.sc.gz",
         "/home/kedz/projects2015/trec2015/data/articles/gold/2012_buenos_aires_rail_disaster/2012-02-22-14.sc.gz",
         "/home/kedz/projects2015/trec2015/data/articles/gold/2012_buenos_aires_rail_disaster/2012-02-22-15.sc.gz",
         ]

matches = cuttsum.judgements.get_merged_dataframe()

#print matches[matches["update id"].apply(lambda x: x.startswith(example_id))]

def si_to_df(si):
    sents = []
    for s, sent in enumerate(si.body.sentences["lingpipe"]):
        sents.append({
            "sent id": s, 
            "sent text": stringify_streamcorpus_sentence(sent).decode("utf-8"),
            "doc id": si.stream_id,
            "update id": si.stream_id + "-" + str(s),
            })
    return pd.DataFrame(sents)


dfs = []
for path in paths:
    with sc.Chunk(path=path, mode="rb", message=corpus.sc_msg()) as chunk:

        for si in chunk:
            df = si_to_df(si)
            df["nuggets"] = df["update id"].apply(
                lambda x: set(matches[matches["update id"] == x]["nugget id"].tolist()))
            dfs.append(df)
df = pd.concat(dfs)
df = df[df["nuggets"].apply(len) > 0]

from sumpy.system import AverageFeatureRankerBase
from sumpy.annotators import LedeMixin, CentroidMixin

class SubmodularSummarizer(LedeMixin, CentroidMixin, AverageFeatureRankerBase):
    pass

print df
system = SubmodularSummarizer(verbose=True)

print system.summarize(df)
