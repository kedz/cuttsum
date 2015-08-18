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
import corenlp
from cuttsum.misc import english_stopwords, event2semsim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

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
                classifiers.append((nugget_id, model[0], model[1], model[2]))

#        def remove_punctuation(text):
#            return re.sub(
#                ur"\p{P}+", "", 
#                text).lower().encode("utf-8")


        def classify_nuggets(df):
            
            sents = [" ".join(lemmas).lower() for lemmas in df["lemmas stopped"].tolist()]
            sets = [set(sent.split(" ")) for sent in sents]         
            
            #sents = map(remove_punctuation, sents)
            nuggets = [set() for sent in sents]
            nugget_probs = [dict() for sent in sents]            

            all_probs = []
            for nugget_id, vec, clf, nugget_lems in classifiers:
                if vec is not None and clf is not None:
                    
                    X = vec.transform(sents).todense()
                    x_cov = [len(nugget_lems.intersection(set(lems))) / float(len(nugget_lems))
                             for lems in sets]

                    x_cov = np.array(x_cov)[:, np.newaxis]
                    X = np.hstack([X, x_cov])
                    P = clf.predict_proba(X)
                    y = np.zeros(X.shape[0], dtype="int32")
                    #y[(P[:,1] > .95) | ((x_cov[:,0] > .75) & (len(nugget_lems) > 1))] = 1
                    for i in np.where(y == 1)[0]:
                        #if X[i].sum() < 3:
                        #if len(sents[i].split(" ")) < 6:
                         #   continue
                        nuggets[i].add(nugget_id.encode("utf-8")) 
                    
                    for i in xrange(len(nugget_probs)):
                            #if P[i,1] > .5:
                            nugget_probs[i][nugget_id.encode("utf-8")] = P[i,1]
                    all_probs.append(P[:,1,np.newaxis])
                #else:
                    #x_cov = [len(nugget_lems.intersection(set(lems))) / float(len(nugget_lems))
                    #         for lems in sets]
                    #x_cov = np.array(x_cov)[:, np.newaxis]
                    #y = np.zeros(x_cov.shape[0], dtype="int32")
                    #y[((x_cov[:,0] > .75) & (len(nugget_lems) > 1))] = 1
                    #for i in np.where(y == 1)[0]:
                    #    nuggets[i].add(nugget_id.encode("utf-8")) 
                    #for i in xrange(len(nugget_probs)):
                    #    if x_cov[i,0] > .5:
                    #        nugget_probs[i][nugget_id.encode("utf-8")] = x_cov[i,0]

            Pp = np.hstack(all_probs)
            conf = np.max(Pp, axis=1)

            return nuggets, conf, nugget_probs
                       
#for sent in sents:
#            sent
#            return 


        return classify_nuggets 


    def get_best_model(self, event, nugget_id):
#        stats_path = self.get_stats_path(event, nugget_id)
#        with open(stats_path, "r") as f:
#            df = pd.read_csv(f, sep="\t")
#        df = df[df["coverage"] >= 0.5]
#        df = df[df["pos prec"] >= 0.7]
#        if len(df) == 0:
#            return None
#        else:
#            df.reset_index(inplace=True)
#            argmax = df.iloc[df["pos prec"].argmax()]
#            model_path = self.get_model_path(event, nugget_id, argmax["name"])
#            clf = joblib.load(model_path)           
#            vec_path = self.get_vectorizer_path(event, nugget_id)
#            vec = joblib.load(vec_path)
#            return (vec, clf)

        model_path = self.get_model_path(event, nugget_id, "gbc")
        vec_path = self.get_vectorizer_path(event, nugget_id)
        if os.path.exists(model_path) and os.path.exists(vec_path):
            vec, lemmas = joblib.load(vec_path)
            clf = joblib.load(model_path)           
            return (vec, clf, lemmas)
        else:
            return None



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
        matches = df[df["query id"] == event.query_id]
        nugget_ids = list(set(matches["nugget id"].tolist()))
        #df = nuggets.filter(lambda x: len(x) > 10)
        #nugget_ids = list(set(df["nugget id"].tolist()))
        nugget_ids.sort()
        units = [i for i, n in enumerate(nugget_ids)
                 if not os.path.exists(self.get_model_path(event, n, "gbc"))]
        return units

    def do_job_unit(self, event, corpus, unit, **kwargs):
        stopwords = english_stopwords()

        ### Preprocessing here. ###
        df = cuttsum.judgements.get_merged_dataframe()
        matches = df[df["query id"] == event.query_id]
        matching_update_ids = set(matches["update id"].tolist())
        #nuggets = matches.groupby("nugget id")
        #thrsh_nuggets = all_nuggets.filter(lambda x: len(x) > 10)

        nugget_ids = list(set(matches["nugget id"].tolist()))

        #nugget_ids = list(set(all_nuggets["nugget id"].tolist()))
        nugget_ids.sort()
        nugget_id = nugget_ids[unit]
        with corenlp.Server(port=9876 + event.query_num * 100 + unit, mem="20G", threads=8, max_message_len=524288,
                annotators=["tokenize", "ssplit", "pos", "lemma"], #, "ner"],
                corenlp_props={
                    "pos.maxlen": 50, "ssplit.eolonly": "true"}) as pipeline:

            if event.query_id.startswith("TS13"):
                updates = cuttsum.judgements.get_2013_updates()
            elif event.query_id.startswith("TS14"):
                updates = cuttsum.judgements.get_2014_sampled_updates()

            updates = updates[updates["query id"] == event.query_id]  
            non_matching_updates = updates[updates["update id"].apply(
                lambda x: x not in matching_update_ids)]
            matching_updates = matches[matches["nugget id"] == nugget_id]
           # if len(matching_updates) == 0:
            #    return 
                #matching_updates = df[df["nugget id"] == nugget_id]

            nugget_text = matching_updates.iloc[0]["nugget text"]
            n_matching = len(matching_updates)
            n_nonmatching = min(n_matching, len(non_matching_updates))
            n_instances = n_matching + n_nonmatching

            semsim = event2semsim(event)
            from nltk.stem.porter import PorterStemmer
            stemmer = PorterStemmer()

            nugget_doc = pipeline.annotate(nugget_text)
            nugget_lems = []
            nugget_stems = []
            for sent in nugget_doc:
                for tok in sent:
                    if unicode(tok).lower() not in stopwords and len(unicode(tok)) < 50:
                        nugget_lems.append(tok.lem.lower())
                    stem = stemmer.stem(unicode(tok).lower())
                    if len(stem) < 50:
                        nugget_stems.append(stem)
            nugget_stems = [u" ".join(nugget_stems)]

            if n_matching <= 10:
                model_dir = self.get_model_dir(event, nugget_id)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
            
                joblib.dump([None, set(nugget_lems)], self.get_vectorizer_path(event, nugget_id), compress=9)
                joblib.dump([], self.get_model_path(event, nugget_id, "gbc"), compress=9)
                return 


            non_matching_updates = non_matching_updates.iloc[
                np.random.permutation(len(non_matching_updates))] 
            non_matching_updates = non_matching_updates.iloc[
                np.arange(n_nonmatching)] 
            #non_matching_updates["text"] = \
            #    non_matching_updates["text"].apply(lambda x: x.lower())

            y = np.zeros(n_instances, dtype="int32")
            y[:n_matching] = 1
            X_string = matching_updates["update text"].tolist()
            X_string += non_matching_updates.head(n_nonmatching)["text"].tolist()
            assert len(X_string) == n_instances

            p = np.random.permutation(n_instances)
            y = y[p]
            X_string = [X_string[i] for i in p]
            print "pipeline start"
            docs = pipeline.annotate_mp(X_string, n_procs=8)
            print "pipeline done"

            lemmas = []
            all_stems = []
            for doc in docs:
                lems = []
                stems = []
                for sent in doc:
                    for tok in sent:
                        if unicode(tok).lower() not in stopwords and len(unicode(tok)) < 50:
                            lems.append(tok.lem.lower())
                        stem = stemmer.stem(unicode(tok).lower())
                        if len(stem) < 50:
                            stems.append(stem)
                #print lems
                lemmas.append(lems)
                all_stems.append(u" ".join(stems))

                        
    # map(
    #                    lambda doc: [str(tok) 
    #                                 for doc in docs
    #                                 for sent in doc
    #                                 for tok in sent

            K = cosine_similarity(
                semsim.transform(all_stems),
                semsim.transform(nugget_stems))

            X_string = [u" ".join(lem) for lem in lemmas]
            vec = TfidfVectorizer(
                input=u"content", stop_words="english", ngram_range=(1,5))
            vec.fit([u" ".join(nugget_lems)] + X_string)
            X = vec.transform(X_string).todense()
            
            nugget_lems = set(nugget_lems)
            x_cov = [len(nugget_lems.intersection(set(lems))) / float(len(nugget_lems))
                     for lems in lemmas]
            x_cov = np.array(x_cov)[:, np.newaxis]
            X = np.hstack([X, x_cov, K, K * x_cov])
            
            
            gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=.1,
                max_depth=8, random_state=0, max_features="log2")
            gbc.fit(X, y)
            model_dir = self.get_model_dir(event, nugget_id)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            joblib.dump([vec, nugget_lems], self.get_vectorizer_path(event, nugget_id), compress=9)
            joblib.dump(gbc, self.get_model_path(event, nugget_id, "gbc"), compress=9)

#
#
#
#        def remove_punctuation(text):
#            return re.sub(
#                ur"\p{P}+", "", 
#                text.decode("utf-8")).lower().encode("utf-8")
#        matching_updates["update text"] = \
#            matching_updates["update text"].apply(remove_punctuation)
# 
#        non_matching_updates = non_matching_updates.iloc[
#            np.random.permutation(len(non_matching_updates))] 
#        non_matching_updates = non_matching_updates.iloc[
#            np.arange(n_nonmatching)] 
#        non_matching_updates["text"] = \
#            non_matching_updates["text"].apply(remove_punctuation)
#
#        y = np.zeros(n_instances, dtype="int32")
#        y[:n_matching] = 1
#        X_string = matching_updates["update text"].tolist()
#        X_string += non_matching_updates.head(n_nonmatching)["text"].tolist()
#        assert len(X_string) == n_instances
#       
#        p = np.random.permutation(n_instances)
#        y = y[p]
#        X_string = [X_string[i] for i in p]
#
#        vec = TfidfVectorizer(
#            input=u"content", stop_words="english", ngram_range=(2,5))
#        X = vec.fit_transform(X_string) 
#
#
#        model_dir = self.get_model_dir(event, nugget_id)
#        if not os.path.exists(model_dir):
#            os.makedirs(model_dir)
#
#        joblib.dump(vec, self.get_vectorizer_path(event, nugget_id), compress=9)
#        joblib.dump(clf, self.get_model_path(event, nugget_id, model_name), compress=9)
#
#        ### Classifier shootout here. ###
#        #prob_thresh = .5
#
#        def prec_scorer(clf, X, y):
#            y_hat = clf.predict(X)
#            #y_hat = np.zeros(X.shape[0], dtype="int32")
#            #y_hat[P[:,1] > prob_thresh] = 1 
#            return precision_score(y, y_hat)
#
#        def coverage_scorer(clf, X, y):
#            y_hat = clf.predict(X)
#            cov = np.sum((y_hat == 1) & (y == 1)) / y[y==1].sum()
#            return cov
#
#
#
#        results = []
#        for c in [.1, 1., 10., 100.]:
#            for pen in ["l1", "l2"]:
#                clf = LogisticRegression(C=c, penalty=pen, class_weight={0:.3, 1:.7})
#                scores = cross_validation.cross_val_score(clf, X, y, scoring=prec_scorer, cv=10)
#                coverages = cross_validation.cross_val_score(clf, X, y, scoring=coverage_scorer, cv=10)
#                clf.fit(X, y)
##                tps = cross_validation.cross_val_score(clf, X, y, scoring=tp_scorer, cv=10)
#                mu, sigma = (scores.mean(), scores.std())
#                avg_cov, std_cov = (coverages.mean(), coverages.std())
##                tp_mu, tp_sigma = (tps.mean(), tps.std())
#
#                model_name = "C={}_pen={}_LogReg".format(c, pen)
#                results.append({
#                    "nugget id": nugget_id, 
#                    "nugget text": nugget_text, 
#                    "matching": n_matching,
#                    "nonmatching": n_nonmatching,                
#                    "name": model_name,
#                    "pos prec": mu, 
#                    "std": sigma,
#                    "coverage": avg_cov,
#                    "coverage std": std_cov,
#                })
#                joblib.dump(clf, self.get_model_path(event, nugget_id, model_name), compress=9)
#
#        for alp in [.1, 1., 10., 100.]:
#            clf = MultinomialNB(alpha=alp, class_prior=[.7, .3])
#            scores = cross_validation.cross_val_score(clf, X, y, scoring=prec_scorer, cv=10)
#            coverages = cross_validation.cross_val_score(clf, X, y, scoring=coverage_scorer, cv=10)
#            clf.fit(X, y)
#            #tps = cross_validation.cross_val_score(clf, X, y, scoring=tp_scorer, cv=10)
#            mu, sigma = (scores.mean(), scores.std())
#            avg_cov, std_cov = (coverages.mean(), coverages.std())
#            #tp_mu, tp_sigma = (tps.mean(), tps.std())
#            model_name = "alpha={}_MultiNB".format(alp)
#            results.append({
#                "nugget id": nugget_id, 
#                "nugget text": nugget_text, 
#                "matching": n_matching,
#                "nonmatching": n_nonmatching,                
#                "name": model_name,
#                "pos prec": mu, 
#                "std": sigma,
#                "coverage": avg_cov,
#                "coverage std": std_cov,
#            })
#            joblib.dump(clf, self.get_model_path(event, nugget_id, model_name), compress=9)
#
#        df = pd.DataFrame(
#            results, columns=["nugget id", "nugget text", "matching", "nonmatching", "name", 
#                              "pos prec", "std", "coverage", "coverage std"])
#        with open(self.get_stats_path(event, nugget_id), "w") as f:
#            df.to_csv(f, index=False, sep="\t")
