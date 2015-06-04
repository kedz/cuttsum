import os
from cuttsum.resources import MultiProcessWorker
import cuttsum.judgements
import pandas as pd
import numpy as np
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


class NuggetClassifier(MultiProcessWorker):
    def __init__(self):
        #Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'nugget-classifiers')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_classifier(self, event):
        path = os.path.join(self.dir_, event.fs_name())
        classifiers = []
        for nugget_id in os.listdir(path):
            model = self.get_best_model(event, nugget_id) 
            if model is not None:            
                classifiers.append((nugget_id, model[0], model[1]))

        def remove_punctuation(text):
            return re.sub(
                ur"\p{P}+", "", 
                text).lower().encode("utf-8")


        def classify_nuggets(sents):
            sents = map(remove_punctuation, sents)
            nuggets = [set() for sent in sents]

            for nugget_id, vec, clf in classifiers:
                X = vec.transform(sents) 
                P = clf.predict_proba(X)
                y = np.zeros(X.shape[0], dtype="int32")
                y[P[:,1] > .7] = 1
                for i in np.where(y == 1)[0]:
                    #if X[i].sum() < 3:
                    #if len(sents[i].split(" ")) < 6:
                     #   continue
                    nuggets[i].add(nugget_id.encode("utf-8")) 
            return nuggets    
                       
#for sent in sents:
#            sent
#            return 


        return classify_nuggets 


    def get_best_model(self, event, nugget_id):
        stats_path = self.get_stats_path(event, nugget_id)
        with open(stats_path, "r") as f:
            df = pd.read_csv(f, sep="\t")
        df = df[df["coverage"] >= 0.5]
        df = df[df["pos prec"] >= 0.7]
        if len(df) == 0:
            return None
        else:
            df.reset_index(inplace=True)
            argmax = df.iloc[df["pos prec"].argmax()]
            model_path = self.get_model_path(event, nugget_id, argmax["name"])
            clf = joblib.load(model_path)           
            vec_path = self.get_vectorizer_path(event, nugget_id)
            vec = joblib.load(vec_path)
            return (vec, clf)




    def get_model_dir(self, event, nugget_id):
        return os.path.join(self.dir_, event.fs_name(), nugget_id) 

    def get_model_path(self, event, nugget_id, model_name):
        return os.path.join(self.dir_, event.fs_name(), nugget_id, "{}.pkl".format(model_name)) 
                
    def get_stats_path(self, event, nugget_id):
        return os.path.join(self.dir_, event.fs_name(), nugget_id, "stats.tsv") 

    def get_vectorizer_path(self, event, nugget_id):
        return os.path.join(self.dir_, event.fs_name(), nugget_id, "vectorizer.pkl")

    def get_job_units(self, event, corpus, **kwargs):
        df = cuttsum.judgements.get_merged_dataframe()
        df = df[df["query id"] == event.query_id]
        nuggets = df.groupby("nugget id")
        df = nuggets.filter(lambda x: len(x) > 10)
        nugget_ids = list(set(df["nugget id"].tolist()))
        nugget_ids.sort()
        units = [i for i, n in enumerate(nugget_ids)]
        return units

    def do_job_unit(self, event, corpus, unit, **kwargs):

        ### Preprocessing here. ###
        df = cuttsum.judgements.get_merged_dataframe()
        matches = df[df["query id"] == event.query_id]

        matching_update_ids = set(matches["update id"].tolist())
        all_nuggets = matches.groupby("nugget id")
        thrsh_nuggets = all_nuggets.filter(lambda x: len(x) > 10)

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

        def remove_punctuation(text):
            return re.sub(
                ur"\p{P}+", "", 
                text.decode("utf-8")).lower().encode("utf-8")
        matching_updates["update text"] = \
            matching_updates["update text"].apply(remove_punctuation)
 
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

        vec = TfidfVectorizer(
            input=u"content", stop_words="english", ngram_range=(2,5))
        X = vec.fit_transform(X_string) 


        model_dir = self.get_model_dir(event, nugget_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        joblib.dump(vec, self.get_vectorizer_path(event, nugget_id), compress=9)

        ### Classifier shootout here. ###
        #prob_thresh = .5

        def prec_scorer(clf, X, y):
            y_hat = clf.predict(X)
            #y_hat = np.zeros(X.shape[0], dtype="int32")
            #y_hat[P[:,1] > prob_thresh] = 1 
            return precision_score(y, y_hat)

        def coverage_scorer(clf, X, y):
            y_hat = clf.predict(X)
            cov = np.sum((y_hat == 1) & (y == 1)) / y[y==1].sum()
            return cov



        results = []
        for c in [.1, 1., 10., 100.]:
            for pen in ["l1", "l2"]:
                clf = LogisticRegression(C=c, penalty=pen, class_weight={0:.3, 1:.7})
                scores = cross_validation.cross_val_score(clf, X, y, scoring=prec_scorer, cv=10)
                coverages = cross_validation.cross_val_score(clf, X, y, scoring=coverage_scorer, cv=10)
                clf.fit(X, y)
#                tps = cross_validation.cross_val_score(clf, X, y, scoring=tp_scorer, cv=10)
                mu, sigma = (scores.mean(), scores.std())
                avg_cov, std_cov = (coverages.mean(), coverages.std())
#                tp_mu, tp_sigma = (tps.mean(), tps.std())

                model_name = "C={}_pen={}_LogReg".format(c, pen)
                results.append({
                    "nugget id": nugget_id, 
                    "nugget text": nugget_text, 
                    "matching": n_matching,
                    "nonmatching": n_nonmatching,                
                    "name": model_name,
                    "pos prec": mu, 
                    "std": sigma,
                    "coverage": avg_cov,
                    "coverage std": std_cov,
                })
                joblib.dump(clf, self.get_model_path(event, nugget_id, model_name), compress=9)

        for alp in [.1, 1., 10., 100.]:
            clf = MultinomialNB(alpha=alp, class_prior=[.7, .3])
            scores = cross_validation.cross_val_score(clf, X, y, scoring=prec_scorer, cv=10)
            coverages = cross_validation.cross_val_score(clf, X, y, scoring=coverage_scorer, cv=10)
            clf.fit(X, y)
            #tps = cross_validation.cross_val_score(clf, X, y, scoring=tp_scorer, cv=10)
            mu, sigma = (scores.mean(), scores.std())
            avg_cov, std_cov = (coverages.mean(), coverages.std())
            #tp_mu, tp_sigma = (tps.mean(), tps.std())
            model_name = "alpha={}_MultiNB".format(alp)
            results.append({
                "nugget id": nugget_id, 
                "nugget text": nugget_text, 
                "matching": n_matching,
                "nonmatching": n_nonmatching,                
                "name": model_name,
                "pos prec": mu, 
                "std": sigma,
                "coverage": avg_cov,
                "coverage std": std_cov,
            })
            joblib.dump(clf, self.get_model_path(event, nugget_id, model_name), compress=9)

        df = pd.DataFrame(
            results, columns=["nugget id", "nugget text", "matching", "nonmatching", "name", 
                              "pos prec", "std", "coverage", "coverage std"])
        with open(self.get_stats_path(event, nugget_id), "w") as f:
            df.to_csv(f, index=False, sep="\t")
