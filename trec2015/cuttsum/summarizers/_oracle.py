import os
from cuttsum.resources import MultiProcessWorker
from cuttsum.pipeline import ArticlesResource
import cuttsum.judgements
import numpy as np
import pandas as pd
import sumpy



class RetrospectiveMonotoneSubmodularOracle(MultiProcessWorker):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u"TREC_DATA", u"."), "system-results",
            "retrospective-monotone-submodular-oracle-summaries")
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_path_prefix(self, event, corpus, extractor, 
            budget, soft_matching):
        return os.path.join(self.dir_, extractor, str(budget), 
            "soft_match" if soft_matching is True else "no_soft_match", 
            corpus.fs_name(), event.fs_name()) 

    def get_job_units(self, event, corpus, **kwargs):
        extractor = kwargs.get("extractor", "gold")
        if extractor == "gold":
            return [0]
        else:
            raise Exception(
                "extractor: {} not implemented!".format(extractor))


    def do_job_unit(self, event, corpus, unit, **kwargs):
        if unit != 0:
            raise Exception("unit of work out of bounds!") 
        extractor = kwargs.get("extractor", "gold")
        soft_match = kwargs.get("soft_match", False)
        budget = kwargs.get("budget", 25)
        
        output_path_prefix = self.get_path_prefix(
            event, corpus, extractor, budget, soft_match)

        ## Set up summarizer ###
        def f_of_A(system, A, V_min_A, e, input_df, ndarray_data):
            return len(
                set([nugget for nuggets in input_df.ix[A, "nuggets"].tolist()
                     for nugget in nuggets]))


        system = sumpy.system.MonotoneSubmodularBasic(f_of_A=f_of_A, k=budget)

        articles = ArticlesResource()
            
        all_matches = cuttsum.judgements.get_merged_dataframe()
        matches = all_matches[all_matches["query id"] == event.query_id]
        
        from cuttsum.misc import si2df

        if soft_match is True:
            from cuttsum.classifiers import NuggetClassifier
            classify_nuggets = NuggetClassifier().get_classifier(event)
            if event.query_id.startswith("TS13"):
                judged = cuttsum.judgements.get_2013_updates() 
                judged = judged[judged["query id"] == event.query_id]
                judged_uids = set(judged["update id"].tolist())
  
        all_df = [] 
        
        for hour, path, si in articles.streamitem_iter(
                event, corpus, extractor):

            df = si2df(si)
            df["nuggets"] = df["update id"].apply(
                lambda x: set(
                    matches[matches["update id"] == x]["nugget id"].tolist()))

            if soft_match is True:
                I = np.where(
                    df["update id"].apply(lambda x: x not in judged_uids))[0]
                
                unjudged = df[
                    df["update id"].apply(lambda x: x not in judged_uids)]
                unjudged_sents = unjudged["sent text"].tolist()
                assert len(unjudged_sents) == I.shape[0]

                df.loc[I, "nuggets"] = classify_nuggets(unjudged_sents)
            

            df = df[df["nuggets"].apply(len) > 0]
            all_df.append(df)

        all_df = pd.concat(all_df)
        all_df.reset_index(inplace=True)

        summary =  system.summarize(all_df)
        F_of_S = len(
            set(n for ns in summary._df["nuggets"].tolist() for n in ns))

        print "F(S)", F_of_S
        print "summary nuggets" 
        sum_nuggets = list(set(
            n for ns in summary._df["nuggets"].tolist() for n in ns))
        sum_nuggets.sort()
        print sum_nuggets
 
        possible_nuggets = list(set(
            n for ns in all_df["nuggets"].tolist() for n in ns))
        possible_nuggets.sort()
        print possible_nuggets
        print len(possible_nuggets) 

        event_nuggets = set(matches["nugget id"].tolist())
        total_nuggets = len(event_nuggets)
        timestamp = int(si.stream_id.split("-")[0])

        output_df = pd.DataFrame(
            [{"Cum. F(S)": F_of_S, 
              "F(S)": F_of_S, 
              "UB no const.": total_nuggets, 
              "budget": budget, 
              "Tot. Updates": len(summary._df), 
              "event title": event.fs_name(), 
              "timestamp": timestamp, 
              "query id": event.query_id},],
            columns=["timestamp", "query id", "event title", "Cum. F(S)", "F(S)", "UB no const.",
                     "Tot. Updates", "budget",])   
        
        parent = os.path.dirname(output_path_prefix)
        if not os.path.exists(parent):
            os.makedirs(parent)

        stats_path = output_path_prefix + ".stats.tsv"
        updates_path = output_path_prefix + ".updates.tsv"
        
        with open(stats_path, "w") as f:
            output_df.to_csv(f, sep="\t", index=False)
        summary._df["sent text"] = summary._df["sent text"].apply(lambda x: x.encode("utf-8"))
        with open(updates_path, "w") as f:
            summary._df[["timestamp", "update id", "sent text"]].sort(["update id"]).to_csv(
                f, sep="\t", index=False)


class MonotoneSubmodularOracle(MultiProcessWorker):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u"TREC_DATA", u"."), "system-results",
            "monotone-submodular-oracle-summaries")
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_path_prefix(self, event, corpus, extractor, budget, soft_matching):
        return os.path.join(self.dir_, extractor, str(budget), 
            "soft_match" if soft_matching is True else "no_soft_match", 
            corpus.fs_name(), event.fs_name()) 

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
        soft_match = kwargs.get("soft_match", False)
        budget = kwargs.get("budget", 25)
        
        output_path_prefix = self.get_path_prefix(event, corpus, extractor, budget, soft_match)
       
        ## Set up summarizer ###
        def f_of_A(system, A, V_min_A, e, input_df, ndarray_data):
            return len(
                set([nugget for nuggets in input_df.ix[A, "nuggets"].tolist() for nugget in nuggets]))
        system = sumpy.system.MonotoneSubmodularBasic(f_of_A=f_of_A, k=budget)
        nugget_cache = set()

        articles = ArticlesResource()
            
        all_matches = cuttsum.judgements.get_merged_dataframe()
        matches = all_matches[all_matches["query id"] == event.query_id]
        
        from cuttsum.misc import si2df

        stats = []
        summary_df = []
        cum_F_of_S = 0
        event_nuggets = set(matches["nugget id"].tolist())
        total_nuggets = len(event_nuggets)
        total_updates = 0

        for hour, path, si in articles.streamitem_iter(
                event, corpus, extractor):

            df = si2df(si)
            df["nuggets"] = df["update id"].apply(
                lambda x: set(matches[matches["update id"] == x]["nugget id"].tolist()))
            
            df = df[df["nuggets"].apply(len) > 0]
            df["nuggets"] = df["nuggets"].apply(lambda x: x.difference(nugget_cache))
            if len(df) == 0:
                continue
            summary =  system.summarize(df)
            summary_nuggets = set(n for ns in summary._df["nuggets"].tolist() for n in ns)
            nugget_cache.update(summary_nuggets)  
            system.k -= len(summary._df)

            F_of_S = len(summary_nuggets)
            cum_F_of_S += F_of_S
            total_updates += len(summary._df)
            timestamp = int(si.stream_id.split("-")[0])

            stats.append({
                "Cum. F(S)": cum_F_of_S, 
                "F(S)": F_of_S, 
                "UB no const.": total_nuggets, 
                "budget": budget, 
                "Tot. Updates": total_updates, 
                "event title": event.fs_name(), 
                "timestamp": timestamp, 
                "query id": event.query_id,
            })
            summary_df.append(summary._df)
            if system.k <= 0:
                print "Budget exceeded!"
                break


        output_df = pd.DataFrame(stats,
            columns=["timestamp", "query id", "event title", "Cum. F(S)", "F(S)", "UB no const.",
                     "Tot. Updates", "budget",])   
        
        parent = os.path.dirname(output_path_prefix)
        if not os.path.exists(parent):
            os.makedirs(parent)

        stats_path = output_path_prefix + ".stats.tsv"
        updates_path = output_path_prefix + ".updates.tsv"
        

        with open(stats_path, "w") as f:
            output_df.to_csv(f, sep="\t", index=False)

        summary_df = pd.concat(summary_df)
        summary_df["sent text"] = summary_df["sent text"].apply(lambda x: x.encode("utf-8"))
        with open(updates_path, "w") as f:
            summary_df[["timestamp", "update id", "sent text"]].sort(["update id"]).to_csv(
                f, sep="\t", index=False)

