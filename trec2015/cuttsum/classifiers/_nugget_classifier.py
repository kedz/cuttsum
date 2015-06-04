import os
from cuttsum.resources import MultiProcessWorker
import cuttsum.judgements
import pandas as pd

class NuggetClassifier(MultiProcessWorker):
    def __init__(self):
        #Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'nugget-classifiers')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_job_units(self, event, corpus, **kwargs):
        df = cuttsum.judgements.get_merged_dataframe()
        df = df[df["query id"] == event.query_id]
        nuggets = df.groupby("nugget id")
        df = nuggets.filter(lambda x: len(x) > 50)
        nugget_ids = list(set(df["nugget id"].tolist()))
        nugget_ids.sort()
        units = [i for i, n in enumerate(nugget_ids)]
        return units

    def do_job_unit(self, event, corpus, unit, **kwargs):

        df = cuttsum.judgements.get_merged_dataframe()

        matches = df[df["query id"] == event.query_id]

        matching_update_ids = set(matches["update id"].tolist())
        all_nuggets = matches.groupby("nugget id")
        thrsh_nuggets = all_nuggets.filter(lambda x: len(x) > 50)

        nugget_ids = list(set(thrsh_nuggets["nugget id"].tolist()))
        nugget_ids.sort()
        nugget_id = nugget_ids[unit]
        
        if event.query_id.startswith("TS13"):
            updates = cuttsum.judgements.get_2013_updates()
        elif event.query_id.startswith("TS14"):
            updates = cuttsum.judgements.get_2014_sampled_updates()

        updates = updates[updates["query id"] == event.query_id]  
        non_matching_updates = updates[updates["update id"].apply(
            lambda x: x not in matching_update_ids)]
        matching_updates = matches[matches["nugget id"] == nugget_id]

        nugget_text = matching_updates.iloc[0]["nugget text"]
        n_matching = len(matching_updates)
        n_nonmatching = min(n_matching, len(non_matching_updates))
        n_instances = n_matching + n_nonmatching
        stats = pd.DataFrame(
            [{"nugget id": nugget_id, 
              "nugget text": nugget_text,
              "matching": n_matching,
              "nonmatching": n_nonmatching
            }],
            columns=[
                "nugget id", "nugget text", "matching", "nonmatching"])
        print stats 
        import numpy as np
        import regex as re

        def remove_punctuation(text):
            return re.sub(
                ur"\p{P}+", "", 
                text.decode("utf-8")).lower().encode("utf-8")
        import string
        #s = "string. With. Punctuation?" # Sample string 
        #out = s.translate(string.maketrans("",""), string.punctuation)
        def filter_text(text):
            return text.translate(
                string.maketrans("", ""), 
                string.punctuation).replace("-", " ")
            #text.replace(".", "")
        matching_updates["update text"] = \
            matching_updates["update text"].apply(remove_punctuation)
        print matching_updates["update text"]
 
        non_matching_updates = non_matching_updates.iloc[
            np.random.permutation(len(non_matching_updates))] 
        non_matching_updates = non_matching_updates.iloc[
            np.arange(n_nonmatching)] 
        non_matching_updates["text"] = \
            non_matching_updates["text"].apply(remove_punctuation)

        y = np.zeros(n_instances, dtype="int32")
        y[:n_matching] = 1
        X_string = matching_updates["update text"].tolist()
        X_string += non_matching_updates.head(n_nonmatching)["text"].tolist()
        assert len(X_string) == n_instances
       
        p = np.random.permutation(n_instances)
        y = y[p]
        X_string = [X_string[i] for i in p]

        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(
            input=u"content", stop_words="english", ngram_range=(1,5))
        X = vec.fit_transform(X_string) 
        from sklearn.linear_model import LogisticRegression
        from sklearn import cross_validation
        clf = LogisticRegression(C=10., penalty="l1")
        scores = cross_validation.cross_val_score(
            clf, X, y, cv=10)
        
        mu, sigma = (scores.mean(), scores.std())
        print "LR Acc. {} (+/- {})".format(mu, sigma)

        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB(alpha=10)

        scores = cross_validation.cross_val_score(
            clf, X, y, cv=10)
        mu, sigma = (scores.mean(), scores.std())
        print "NB Acc. {} (+/- {})".format(mu, sigma)


#(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0)
        #fltr = nuggets.size() > 50
        #print nuggets[fltr > 50]
        #print nuggets_fltr
