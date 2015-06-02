import streamcorpus as sc
import cuttsum.judgements
import cuttsum.events
import cuttsum.corpora
from cuttsum.misc import stringify_streamcorpus_sentence

event = cuttsum.events.get_events(by_query_ids=["TS13.1"])[0]
corpus = cuttsum.corpora.EnglishAndUnknown2013()

example_path = "/scratch/t-chkedz/trec-data/articles/gold/2012_buenos_aires_rail_disaster/2012-02-23-01.sc.gz"
example_id = "1329959700-18d497cf08e3500f195066be60e6a201"

matches = cuttsum.judgements.get_merged_dataframe()

from goose import Goose, Configuration

config = Configuration()
config.enable_image_fetching = False
g = Goose(config)

def si_to_df(si):
    sents = []
    for s, sent in enumerate(si.body.sentences["lingpipe"]):
        sents.append(            


with sc.Chunk(path=example_path, mode="rb", message=corpus.sc_msg()) as chunk:


    for si in chunk:
        if si.stream_id == example_id:
            for s, sent in enumerate(si.body.sentences["lingpipe"]):
                uid = example_id+"-"+str(s)
                sent_matches = matches[matches["update id"] == uid]
                if len(sent_matches) > 0:
                    print s, stringify_streamcorpus_sentence(sent)
                    for ntext in sent_matches["nugget id"].tolist():
                        print "\t", ntext
            article = g.extract(raw_html=si.body.clean_html)
            print article.cleaned_text
    


