import cuttsum.events
import cuttsum.judgements
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import corenlp
from cuttsum.misc import english_stopwords

dom2type = {
    "accidents": set(["accident"]),
    "natural-disasters": set(["earthquake", "storm", "impact event"]),
    "social-unrest": set(["protest", "riot"]),
    "terrorism": set(["shooting", "bombing", "conflict", "hostage"]),
}



def main2():
    events = cuttsum.events.get_events()
    df = cuttsum.judgements.get_merged_dataframe()
    stopwords = english_stopwords()

    with corenlp.Server(port=9876, mem="20G", threads=8, max_message_len=524288,
            annotators=["tokenize", "ssplit", "pos", "lemma"], #, "ner"],
            corenlp_props={
                "pos.maxlen": 50, "ssplit.eolonly": "true"}) as pipeline:
        
   
        for event in events[8:9]:

            matches = df[df["query id"] == event.query_id]

            matching_update_ids = set(matches["update id"].tolist())
            all_nuggets = matches.groupby("nugget id")
            thrsh_nuggets = all_nuggets.filter(lambda x: len(x) <= 10)

            nugget_ids = list(set(thrsh_nuggets["nugget id"].tolist()))
            nugget_ids.sort()

            for nugget_id in nugget_ids:
                
                if event.query_id.startswith("TS13"):
                    updates = cuttsum.judgements.get_2013_updates()
                elif event.query_id.startswith("TS14"):
                    updates = cuttsum.judgements.get_2014_sampled_updates()

                updates = updates[updates["query id"] == event.query_id]  
                non_matching_updates = updates[updates["update id"].apply(
                    lambda x: x not in matching_update_ids)]
                matching_updates = matches[matches["nugget id"] == nugget_id]

                nugget_text = matching_updates.iloc[0]["nugget text"]
                print nugget_text
                n_matching = len(matching_updates)
                n_nonmatching = len(non_matching_updates)
                n_instances = n_matching + n_nonmatching

                #matching_updates["update text"] = \
                #    matching_updates["update text"].apply(lambda x: x.lower())
         
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
                nugget_doc = pipeline.annotate(nugget_text)
                print "pipeline done"

                lemmas = []
                for doc in docs:
                    lems = []
                    for sent in doc:
                        for tok in sent:
                            if unicode(tok).lower() not in stopwords and len(unicode(tok)) < 50:
                                lems.append(tok.lem.lower())
                    #print lems
                    lemmas.append(set(lems))

                nugget_lems = []
                for sent in nugget_doc:
                    for tok in sent:
                        if unicode(tok).lower() not in stopwords and len(unicode(tok)) < 50:
                            nugget_lems.append(tok.lem.lower())

                nugget_lems = set(nugget_lems)
                n_lems = float(len(nugget_lems))
                if n_lems == 1: 
                    print
                    continue 
                for i in xrange(n_instances):
                    if len(lemmas[i]) > 50: continue 
                    cov = len(nugget_lems.intersection(lemmas[i])) / n_lems
                    if cov > .75:
                        if isinstance(nugget_text, str):
                            print nugget_text
                        else:
                            print nugget_text.encode("utf-8")

                        if isinstance(X_string[i], str):
                            print y[i], X_string[i]
                        else:
                            print y[i], X_string[i].encode("utf-8")
                print
def main():
    events = cuttsum.events.get_events()
    df = cuttsum.judgements.get_merged_dataframe()
    stopwords = english_stopwords()

    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    import os
    from sklearn.externals import joblib
    wtmf_models = {}
    wtmf_models["accidents"] = joblib.load(os.getenv("TREC_DATA") + "/semsim/accidents.norm-stem.lam20.000.pkl")
    wtmf_models["social-unrest"] = joblib.load(os.getenv("TREC_DATA") + "/semsim/social-unrest.norm-stem.lam1.000.pkl")
    wtmf_models["terrorism"] = joblib.load(os.getenv("TREC_DATA") + "/semsim/terrorism.norm-stem.lam10.000.pkl")
    wtmf_models["natural-disasters"] = joblib.load(os.getenv("TREC_DATA") + "/semsim/natural-disasters.norm-stem.lam20.000.pkl")

    all_acc = []
    all_aug_acc = []

    with corenlp.Server(port=9876, mem="20G", threads=8, max_message_len=524288,
            annotators=["tokenize", "ssplit", "pos", "lemma"], #, "ner"],
            corenlp_props={
                "pos.maxlen": 50, "ssplit.eolonly": "true"}) as pipeline:
        
   
        for event in events:
            if event.query_num == 7: continue 
            if event.query_num > 25: continue 

            if event.type in dom2type["natural-disasters"]: 
                wtmf_vec = wtmf_models["natural-disasters"]

            if event.type in dom2type["accidents"]:
                wtmf_vec = wtmf_models["accidents"]
            
            if event.type in dom2type["social-unrest"]:
                wtmf_vec = wtmf_models["social-unrest"]
 
            if event.type in dom2type["terrorism"]:
                wtmf_vec = wtmf_models["terrorism"]
            
            matches = df[df["query id"] == event.query_id]

            matching_update_ids = set(matches["update id"].tolist())
            all_nuggets = matches.groupby("nugget id")
            thrsh_nuggets = all_nuggets.filter(lambda x: len(x) > 10)

            nugget_ids = list(set(thrsh_nuggets["nugget id"].tolist()))
            #nugget_ids.sort()

            for num_nug, nugget_id in enumerate(nugget_ids):
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

                #matching_updates["update text"] = \
                #    matching_updates["update text"].apply(lambda x: x.lower())
         
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
                nugget_doc = pipeline.annotate(nugget_text)
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
                            stems.append(stemmer.stem(unicode(tok).lower()))
                    #print lems
                    lemmas.append(lems)
                    all_stems.append(u" ".join(stems))

                

                nugget_lems = []
                nugget_stems = []
                for sent in nugget_doc:
                    for tok in sent:
                        if unicode(tok).lower() not in stopwords and len(unicode(tok)) < 50:
                            nugget_lems.append(tok.lem.lower())
                        nugget_stems.append(stemmer.stem(unicode(tok).lower()))
                nugget_stems = [u" ".join(nugget_stems)]        
# map(
#                    lambda doc: [str(tok) 
#                                 for doc in docs
#                                 for sent in doc
#                                 for tok in sent

                X_string = [u" ".join(lem) for lem in lemmas]
                vec = TfidfVectorizer(
                    input=u"content", stop_words="english", ngram_range=(1,5))
                vec.fit([u" ".join(nugget_lems)] + X_string)
                X = vec.transform(X_string).todense()
                
                nugget_lems = set(nugget_lems)
                x_cov = [len(nugget_lems.intersection(set(lems))) / float(len(nugget_lems))
                         for lems in lemmas]
                x_cov = np.array(x_cov)[:, np.newaxis]
                X = np.hstack([X, x_cov])
                #print X[:, -1]
                #X_nug = vec.transform([u" ".join(nugget_lems)]).todense()
                
                from sklearn.cross_validation import StratifiedKFold
                from sklearn.metrics import classification_report
                from sklearn.ensemble import GradientBoostingClassifier
                
                gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=.1,
                    max_depth=8, random_state=0, max_features="log2")

                #scores = cross_validation.cross_val_score(gbc, X, y, cv=10)
                #print scores.mean()

                K = cosine_similarity(wtmf_vec.transform(all_stems), wtmf_vec.transform(nugget_stems))

                X_aug = np.hstack([X, K, K * x_cov])

                scores = []
                aug_scores = []
                print event.fs_name(), nugget_text
                for train_index, test_index in StratifiedKFold(y, n_folds=10): 
                    X_train = X[train_index]
                    y_train = y[train_index]
                    X_test = X[test_index]
                    y_test = y[test_index]
                    gbc.fit(X_train, y_train)
                    score = gbc.score(X_test, y_test)
                    X_aug_train = X_aug[train_index]
                    y_train = y[train_index]
                    X_aug_test = X_aug[test_index]
                    y_test = y[test_index]
                    gbc.fit(X_aug_train, y_train)
                    score_aug = gbc.score(X_aug_test, y_test)



                    print score, score_aug
                    scores.append(score)
                    aug_scores.append(score_aug)
                print "mean", np.mean(scores), np.mean(aug_scores)
                all_aug_acc.append(np.mean(aug_scores))
                all_acc.append(np.mean(scores))

                print classification_report(y_test, gbc.predict(X_aug_test))
                y_pred = gbc.predict(X_aug)
                for i, c in enumerate(y_pred):
                    if c == 0 and y[i] == 1:
                        print nugget_text #.encode("utf-8")
                        print X_string[i] #.encode("utf-8")

                print
                print "False positives"
                for i, c in enumerate(y_pred):
                    if c == 1 and y[i] == 0:
                        print nugget_text #.encode("utf-8")
                        print X_string[i] #.encode("utf-8")


    #        model_dir = self.get_model_dir(event, nugget_id)
    #        if not os.path.exists(model_dir):
    #            os.makedirs(model_dir)

     #       joblib.dump(vec, self.get_vectorizer_path(event, nugget_id), compress=9)

            ### Classifier shootout here. ###
            #prob_thresh = .5
    print "Macro avg acc", np.mean(all_acc), np.mean(all_aug_acc)


main()

