import os
from cuttsum.resources import MultiProcessWorker
import cuttsum.judgements
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import corenlp
from cuttsum.misc import english_stopwords
from cuttsum.pipeline import InputStreamResource


class NuggetRegressor(MultiProcessWorker):
    cols = [
        "BASIC length", #"BASIC char length",
        "BASIC doc position", "BASIC all caps ratio",
        "BASIC upper ratio", 
#    "BASIC lower ratio",
#    "BASIC punc ratio", 
        "BASIC person ratio",
        "BASIC location ratio",
        "BASIC organization ratio", "BASIC date ratio",
        "BASIC time ratio", "BASIC duration ratio",
        "BASIC number ratio", "BASIC ordinal ratio",
        "BASIC percent ratio", 
        "BASIC money ratio",
#    "BASIC set ratio", "BASIC misc ratio"
#]

#query_bw_cols = [
 #       "Q_sent_query_cov",
 #   "Q_sent_syn_cov",
 #   "Q_sent_hyper_cov",
 #   "Q_sent_hypo_cov",
#]#

#query_fw_cols = [
        "Q_query_sent_cov",
        "Q_syn_sent_cov",
        "Q_hyper_sent_cov",
        "Q_hypo_sent_cov",
#]

#lm_cols = [
        "LM domain avg lp",
        "LM gw avg lp",

#sum_cols = [
        "SUM_sbasic_sum",
        "SUM_sbasic_amean",
#    "SUM_sbasic_max",
        "SUM_novelty_gmean",
        "SUM_novelty_amean",
#    "SUM_novelty_max",
        "SUM_centrality",
        "SUM_pagerank",

        "SUM_sem_novelty_gmean",
        "SUM_sem_novelty_amean",
#    "SUM_novelty_max",
        "SUM_sem_centrality",
        "SUM_sem_pagerank",
 #]

#stream_cols = [
        "STREAM_sbasic_sum",
        "STREAM_sbasic_amean",
        "STREAM_sbasic_max",
        "STREAM_per_prob_sum",
        "STREAM_per_prob_max",
        "STREAM_per_prob_amean",
        "STREAM_loc_prob_sum",
        "STREAM_loc_prob_max",
        "STREAM_loc_prob_amean",
        "STREAM_org_prob_sum",
        "STREAM_org_prob_max",
        "STREAM_org_prob_amean",
        "STREAM_nt_prob_sum",
        "STREAM_nt_prob_max",
        "STREAM_nt_prob_amean",
    ]



    def __init__(self):
        #Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'nugget-regressors')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)


    def predict(self, event, df):

        gbc = joblib.load(self.get_model_path(event))
        return gbc.predict(df[self.cols].values)

    def get_model_dir(self, event):
        return os.path.join(self.dir_, event.fs_name()) 

    def get_model_path(self, event):
        return os.path.join(self.dir_, event.fs_name(), "gbc.pkl") 


    def get_job_units(self, event, corpus, **kwargs):
        print self.get_model_path(event)
        if not os.path.exists(self.get_model_path(event)):
            return [0]
        else:
            return []

    def do_job_unit(self, event, corpus, unit, **kwargs):
        assert unit == 0

        extractor = kwargs.get('extractor', "goose")
        thresh = kwargs.get('thresh', .8)
        delay = kwargs.get('delay', None)
        topk = kwargs.get('topk', 20)

        train_events = [e for e in cuttsum.events.get_events()
                        if e.query_num not in set([event.query_num, 7])]
        res = InputStreamResource()

        y = []
        X = []
        for train_event in train_events:

            y_e = []
            X_e = []

            istream = res.get_dataframes(
                train_event,
                cuttsum.corpora.get_raw_corpus(train_event), 
                extractor, thresh, delay, topk)
            for df in istream:

                selector = (df["n conf"] == 1) & (df["nugget probs"].apply(len) == 0)
                df.loc[selector, "nugget probs"] = \
                    df.loc[selector, "nuggets"].apply(lambda x: {n:1 for n in x})


                df["probs"] = df["nugget probs"].apply(lambda x: [val for key, val in x.items()] +[0])
                df["probs"] = df["probs"].apply(lambda x: np.max(x))
                df.loc[(df["n conf"] == 1) & (df["nuggets"].apply(len) == 0), "probs"] = 0
                y_t = df["probs"].values
                y_t = y_t[:, np.newaxis]
                y_e.append(y_t)
                X_t = df[self.cols].values
                X_e.append(X_t)

            y_e = np.vstack(y_e)
            y.append(y_e)
            X_e = np.vstack(X_e)
            X.append(X_e)

 #       print "WARNING NOT USING 2014 EVENTS"
        X = np.vstack(X)
        y = np.vstack(y)

        gbc = GradientBoostingRegressor(
            n_estimators=100, learning_rate=1.,
            max_depth=3, random_state=0)
        print "fitting", event
        gbc.fit(X, y.ravel())
        print event, "SCORE", gbc.score(X, y.ravel())
        
        model_dir = self.get_model_dir(event)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        joblib.dump(gbc, self.get_model_path(event), compress=9)

