import os
from cuttsum.resources import MultiProcessWorker
from cuttsum.pipeline import ArticlesResource
import cuttsum.judgements


class MonotoneSubmodularOracle(MultiProcessWorker):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u"TREC_DATA", u"."), 
            "monotone-submodular-oracle-summaries")
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_job_units(self, event, corpus, **kwargs):
        extractor = kwargs.get("extractor", "gold")
        if extractor == "gold":
            return [0]
        else:
            raise Exception("extractor: {} not implemented!".format(extractor))


    def do_job_unit(self, event, corpus, unit, **kwargs):
        if unit != 0:
            raise Exception("unit of work out of bounds!") 
        extractor = kwargs.get("extractor", "gold")
        articles = ArticlesResource()
            
        all_matches = cuttsum.judgements.get_merged_dataframe()
        matches = all_matches[all_matches["query id"] == event.query_id]

        
        for hour, path, si in articles.streamitem_iter(
                event, corpus, extractor):
            print hour, path, si.stream_id,
            doc_matches = matches[matches["update id"].apply(lambda x: x.startswith(si.stream_id))]
            print len(set(doc_matches["update id"].tolist()))
        #for hour in event.list_event_hours():
                                
            
