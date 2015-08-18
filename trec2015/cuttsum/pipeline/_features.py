from cuttsum.resources import MultiProcessWorker
from cuttsum.pipeline import DedupedArticlesResource
from cuttsum.misc import english_stopwords, event2lm_name
import cuttsum.srilm
import gzip
import corenlp as cnlp
import os
import re
from itertools import izip
import numpy as np
from nltk.corpus import wordnet as wn
import pandas as pd
pd.set_option('display.width', 1000)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import gmean
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict


class SentenceFeaturesResource(MultiProcessWorker):
    def __init__(self):
        #Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'sentence-features')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def requires_services(self, event, corpus, **kwargs):
        return ["corenlp", "gigaword-lm", event2lm_name(event)]    

    def get_path(self, event, corpus, extractor, threshold):
        return os.path.join(self.dir_,
            corpus.fs_name(),
            extractor,
            str(threshold),
            event.fs_name() + ".tsv.gz")

    def get_dataframe(self, event, corpus, extractor, threshold):
        path = self.get_path(event, corpus, extractor, threshold)
        with gzip.open(path, "r") as f:
            return pd.read_csv(f, converters={
                "tokens": eval, 
                "tokens stopped": eval, 
                "lemmas stopped": eval,
                "stems": eval}, sep="\t") #, engine="python")


    def get_job_units(self, event, corpus, **kwargs):
        overwrite = kwargs.get("overwrite", False)
        if overwrite is True: return [0]
         
        thresh = kwargs.get("dedupe-sim-threshold", .8)
        extractor = kwargs.get("extractor", "goose")
        path = self.get_path(
            event, corpus, extractor, thresh)
        if not os.path.exists(path):
            return [0]
        else:
            return []

    def do_job_unit(self, event, corpus, unit, **kwargs):
        if unit != 0:
            raise Exception("Job unit {} out of range".format(unit))

        service_configs = kwargs.get("service-configs", {})
        cnlp_configs = service_configs.get("corenlp", {})
        cnlp_port = int(cnlp_configs.get("port", 9999))

        domain_lm_config = service_configs[event2lm_name(event)]
        domain_lm_port = int(domain_lm_config["port"])      
        domain_lm_order = int(domain_lm_config.get("order", 3))	
        gw_lm_config = service_configs["gigaword-lm"]  
        gw_lm_port = int(gw_lm_config["port"])        
        gw_lm_order = int(gw_lm_config.get("order", 3))	


        thresh = kwargs.get("dedupe-sim-threshold", .8)
        extractor = kwargs.get("extractor", "goose")

        res = DedupedArticlesResource()
        dfiter = res.dataframe_iter(
            event, corpus, extractor, include_matches=None, 
            threshold=thresh)

        domain_lm = cuttsum.srilm.Client(domain_lm_port, domain_lm_order, True)
        gw_lm = cuttsum.srilm.Client(gw_lm_port, gw_lm_order, True)
        cnlp_client = cnlp.client.CoreNLPClient(port=cnlp_port)

        def make_query_synsets():
            synonyms = []
            hypernyms = []
            hyponyms = [] 
            print event.type.split(' ')[0]
            for synset in wn.synsets(event.type.split(' ')[0]):
                synonyms.extend(
                    [lemma.name().lower().replace(u'_', u' ').encode(u'utf-8')
                     for lemma in synset.lemmas()])

                hypernyms.extend(
                    [lemma.name().lower().replace(u'_', u' ').encode(u'utf-8')
                     for synset in synset.hypernyms()
                     for lemma in synset.lemmas()])

                hyponyms.extend(
                    [lemma.name().lower().replace(u'_', u' ').encode(u'utf-8')
                     for synset in synset.hyponyms()
                     for lemma in synset.lemmas()])
            print hypernyms
            print hyponyms
            print synonyms
            return set(synonyms), set(hypernyms), set(hyponyms)

        def heal_text(sent_text):
            sent_text = re.sub(
                ur"[A-Z ]+, [A-Z][a-z ]+\( [A-Z]+ \) [-\u2014_]+ ", 
                r"", sent_text)
            sent_text = re.sub(
                ur"^.*?[A-Z ]+, [A-Z][a-z]+ [-\u2014_]+ ", 
                r"", sent_text)
            sent_text = re.sub(
                ur"^.*?[A-Z ]+\([^\)]+\) [-\u2014_]+ ", 
                r"", sent_text)
            sent_text = re.sub(
                ur"^.*?[A-Z]+ +[-\u2014_]+ ", 
                r"", sent_text)
            
            sent_text = re.sub(r"\([^)]+\)", r" ", sent_text)
            sent_text = re.sub(ur"^ *[-\u2014_]+", r"", sent_text)
            sent_text = re.sub(u" ([,.;?!]+)([\"\u201c\u201d'])", r"\1\2", sent_text)
            sent_text = re.sub(r" ([:-]) ", r"\1", sent_text)
            sent_text = re.sub(r"([^\d]\d{1,3}) , (\d\d\d)([^\d]|$)", r"\1,\2\3", sent_text)
            sent_text = re.sub(r"^(\d{1,3}) , (\d\d\d)([^\d]|$)", r"\1,\2\3", sent_text)
            sent_text = re.sub(ur" ('|\u2019) ([a-z]|ll|ve|re)( |$)", r"\1\2 ", sent_text)
            sent_text = re.sub(r" ([',.;?!]+) ", r"\1 ", sent_text)
            sent_text = re.sub(r" ([',.;?!]+)$", r"\1", sent_text)

            sent_text = re.sub(r"(\d\.) (\d)", r"\1\2", sent_text)
            sent_text = re.sub(r"(a|p)\. m\.", r"\1.m.", sent_text)
            sent_text = re.sub(r"U\. (S|N)\.", r"U.\1.", sent_text)

            sent_text = re.sub(
                ur"\u201c ([^\s])", 
                ur"\u201c\1", sent_text)
            sent_text = re.sub(
                ur"([^\s]) \u201d", 
                ur"\1\u201d", sent_text)
            sent_text = re.sub(
                ur"\u2018 ([^\s])", 
                ur"\u2018\1", sent_text)
            sent_text = re.sub(
                ur"([^\s]) \u2019", 
                ur"\1\u2019", sent_text)

            sent_text = re.sub(
                ur"\u00e2", 
                ur"'", sent_text)
            sent_text = re.sub(
                r"^Photo:Reuters|^Photo:AP", 
                r"", sent_text)
            sent_text = sent_text.replace("\n", " ")

            return sent_text.encode("utf-8")

        def get_number_feats(sent):
            feats = []
            for tok in sent:
                if tok.ne == "NUMBER" and tok.nne is not None:
                    for chain in get_dep_chain(tok, sent, 0):
                        feat = [tok.nne] + [elem[1].lem for elem in chain]
                        feats.append(feat)
            return feats
            

        def get_dep_chain(tok, sent, depth):
            chains = []
            if depth > 2:
                return chains
            for p in sent.dep2govs[tok]:
                if p[1].is_noun():
                    for chain in get_dep_chain(p[1], sent, depth + 1):
                        chains.append([p] + chain)
                elif p[1]:
                    chains.append([p])
            return chains
        
        import unicodedata as u
        P=''.join(unichr(i) for i in range(65536) if u.category(unichr(i))[0]=='P')
        P = re.escape(P)
        punc_patt = re.compile("[" + P + "]")

        from collections import defaultdict
        stopwords = english_stopwords()
        mention_counts = defaultdict(int)
        total_mentions = 0

        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()

        synonyms, hypernyms, hyponyms = make_query_synsets()

        path = self.get_path(
            event, corpus, extractor, thresh)
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname): os.makedirs(dirname)

        meta_cols = ["update id", "stream id", "sent id", "timestamp", 
            "pretty text", "tokens", "lemmas", "stems", "pos", "ne", 
            "tokens stopped", "lemmas stopped"]

        basic_cols = ["BASIC length", "BASIC char length", 
            "BASIC doc position", "BASIC all caps ratio", 
            "BASIC upper ratio", "BASIC lower ratio",
            "BASIC punc ratio", "BASIC person ratio", 
            "BASIC location ratio",
            "BASIC organization ratio", "BASIC date ratio", 
            "BASIC time ratio", "BASIC duration ratio",
            "BASIC number ratio", "BASIC ordinal ratio",
            "BASIC percent ratio", "BASIC money ratio", 
            "BASIC set ratio", "BASIC misc ratio"]
       

        lm_cols = ["LM domain lp", "LM domain avg lp",
                   "LM gw lp", "LM gw avg lp"]

        query_cols = [
            "Q_query_sent_cov",
            "Q_sent_query_cov",
            "Q_syn_sent_cov",
            "Q_sent_syn_cov",
            "Q_hyper_sent_cov",
            "Q_sent_hyper_cov",
            "Q_hypo_sent_cov",
            "Q_sent_hypo_cov",
        ]

        sum_cols = [
            "SUM_sbasic_sum",
            "SUM_sbasic_amean",
            "SUM_sbasic_max",
            "SUM_novelty_gmean",
            "SUM_novelty_amean",
            "SUM_novelty_max",
            "SUM_centrality",
            "SUM_pagerank",
            "SUM_sem_novelty_gmean",
            "SUM_sem_novelty_amean",
            "SUM_sem_novelty_max",
            "SUM_sem_centrality",
            "SUM_sem_pagerank",
        ]
    
        stream_cols = [
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


 
        all_cols = meta_cols + basic_cols + query_cols + lm_cols + sum_cols + stream_cols
        
        stream_uni_counts = defaultdict(int)
        stream_per_counts = defaultdict(int)
        stream_loc_counts = defaultdict(int)
        stream_org_counts = defaultdict(int)
        stream_nt_counts = defaultdict(int)

        with gzip.open(path, "w") as f:
            f.write("\t".join(all_cols) + "\n")
            for df in dfiter:
                #df["lm"] = df["sent text"].apply(lambda x: lm.sentence_log_prob(x.encode("utf-8"))[1])
                df["pretty text"] = df["sent text"].apply(heal_text)
                df = df[df["pretty text"].apply(lambda x: len(x.strip())) > 0]
                df = df[df["pretty text"].apply(lambda x: len(x.split(" "))) < 200]
                df = df.reset_index(drop=True)
                if len(df) == 0:
                    print "skipping"
                    continue
                doc_text = "\n".join(df["pretty text"].tolist())
                
                doc = cnlp_client.annotate(doc_text)
                df["tokens"] = map(lambda sent: [str(tok) for tok in sent],
                                   doc)
                df["lemmas"] = map(lambda sent: [tok.lem.encode("utf-8") 
                                                 for tok in sent],
                                   doc)

                df["stems"] = map(lambda sent: 
                    [stemmer.stem(unicode(tok).lower()) for tok in sent], doc)
                df["pos"] = map(lambda sent: [tok.pos for tok in sent],
                                doc)
                
                df["ne"] = map(lambda sent: [tok.ne for tok in sent],
                               doc)
                
                df["tokens stopped"] = map(
                    lambda sent: [str(tok) for tok in sent
                                  if unicode(tok).lower() not in stopwords \
                                      and len(unicode(tok)) < 50],
                    doc)
                df["lemmas stopped"] = map(
                    lambda sent: [tok.lem.lower().encode("utf-8") for tok in sent
                                  if unicode(tok).lower() not in stopwords \
                                      and len(unicode(tok)) < 50],
                    doc)
                
                df["num tuples"] = [get_number_feats(sent) for sent in doc]
                ### Basic Features ###

                df["BASIC length"] = df["lemmas stopped"].apply(len)
                df["BASIC doc position"] = df.index.values + 1
                
                df = df[df["BASIC length"] > 0]
                df = df.reset_index(drop=True)

                df["BASIC char length"] = df["pretty text"].apply(
                    lambda x: len(x.replace(" ", "")))       
        
                df["BASIC upper ratio"] = df["pretty text"].apply(
                    lambda x: len(re.findall("[A-Z]", x))) \
                    / df["BASIC char length"].apply(lambda x: float(max(x, 1)))

                df[ "BASIC lower ratio"] = df["pretty text"].apply(
                    lambda x: len(re.findall("[a-z]", x))) \
                    / df["BASIC char length"].apply(lambda x: float(max(x, 1)))

                df["BASIC punc ratio"] = df["pretty text"].apply(
                    lambda x: len(re.findall(punc_patt, x))) \
                    / df["BASIC char length"].apply(lambda x: float(max(x, 1)))
                df["BASIC all caps ratio"] = df["tokens stopped"].apply(
                    lambda x: np.sum([1 if re.match("^[A-Z]+$", xi) else 0 
                                      for xi in x])) \
                    / df["BASIC length"].apply(float)
               
                df["BASIC person ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "PERSON" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 

                df["BASIC location ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "LOCATION" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 
 
                df["BASIC organization ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "ORGANIZATION" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 
 
                df["BASIC date ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "DATE" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 

                df["BASIC time ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "TIME" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 

                df["BASIC duration ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "DURATION" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 

                df["BASIC number ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "NUMBER" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 

                df["BASIC ordinal ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "ORDINAL" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 

                df["BASIC percent ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "PERCENT" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 

                df["BASIC money ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "MONEY" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 

                df["BASIC set ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "SET" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 

                df["BASIC misc ratio"] = df["ne"].apply(
                    lambda x: np.sum([1 if xi == "MISC" else 0
                                      for xi in x])) \
                    / df["BASIC length"].apply(float) 

                ### Language Model Features ###                

                dm_probs = df["lemmas"].apply(
                    lambda x: domain_lm.sentence_log_prob(
                        " ".join([xi.decode("utf-8").lower().encode("utf-8") 
                                  for xi in x if len(xi) < 50])))
                dm_log_probs = [lp for lp, avg_lp in dm_probs.tolist()]
                dm_avg_log_probs = [avg_lp for lp, avg_lp in dm_probs.tolist()]
                df["LM domain lp"] = dm_log_probs
                df["LM domain avg lp"] = dm_avg_log_probs
                gw_probs = df["lemmas"].apply(
                    lambda x: gw_lm.sentence_log_prob(
                        " ".join([xi.decode("utf-8").lower().encode("utf-8")
                                  for xi in x if len(xi) < 50])))
                gw_log_probs = [lp for lp, avg_lp in gw_probs.tolist()]
                gw_avg_log_probs = [avg_lp for lp, avg_lp in gw_probs.tolist()]
                df["LM gw lp"] = gw_log_probs
                df["LM gw avg lp"] = gw_avg_log_probs


                ### Query Features ###

                self.compute_query_features(df, 
                    set([q.lower() for q in event.query]),
                    synonyms, hypernyms, hyponyms)


                ### Single Doc Summarization Features ###
                
                counts = []
                doc_counts = defaultdict(int)
                for lemmas in df["lemmas stopped"].tolist():
                    counts_i = {}
                    for lem in lemmas:
                        counts_i[lem.lower()] = counts_i.get(lem.lower(), 0) + 1
                        doc_counts[lem.lower()] += 1
                    doc_counts["__TOTAL__"] += len(lemmas)
                    counts.append(counts_i)
                doc_counts["__TOTAL__"] *= 1.
                doc_uni = {key: val / doc_counts["__TOTAL__"] 
                           for key, val in doc_counts.items() 

                           if key != "__TOTAL__"}

                sum_probs = []
                amean_probs = []
                max_probs = []
                for lemmas in df["lemmas stopped"].tolist():
                    probs = [doc_uni[lem.lower()] for lem in lemmas]
                    sum_probs.append(np.sum(probs))
                    amean_probs.append(np.mean(probs))
                    max_probs.append(np.max(probs))

                df["SUM_sbasic_sum"] = sum_probs
                df["SUM_sbasic_amean"] = amean_probs
                df["SUM_sbasic_max"] = max_probs

                tfidfer = TfidfTransformer()
                vec = DictVectorizer()
                X = vec.fit_transform(counts)
                X = tfidfer.fit_transform(X)
        
                ctrd = X.mean(axis=0)
                K = cosine_similarity(ctrd, X).ravel()
                I = K.argsort()[::-1]
                R = np.array([[i, r + 1] for r, i in enumerate(I)])
                R = R[R[:,0].argsort()]
                df["SUM_centrality"] = R[:,1]
                semsim = event2semsim(event)

                L = semsim.transform(df["stems"].apply(lambda x: ' '.join(x)).tolist())
                ctrd_l = L.mean(axis=0)
                K_L = cosine_similarity(ctrd_l, L).ravel()                
                I_L = K_L.argsort()[::-1]
                R_L = np.array([[i, r + 1] for r, i in enumerate(I_L)])
                R_L = R_L[R_L[:, 0].argsort()]
                df["SUM_sem_centrality"] = R_L[:,1]

                K = cosine_similarity(X)
                M = np.zeros_like(K)
                M[np.diag_indices(K.shape[0])] = 1
                Km = np.ma.masked_array(K, M)
                D = 1 - Km
        
                novelty_amean = D.mean(axis=1)
                novelty_max = D.max(axis=1)
                novelty_gmean = gmean(D, axis=1)

                df["SUM_novelty_amean"] = novelty_amean
                df["SUM_novelty_max"] = novelty_max
                df["SUM_novelty_gmean"] = novelty_gmean

                K_L = cosine_similarity(L)
                M_L = np.zeros_like(K)
                M_L[np.diag_indices(K_L.shape[0])] = 1
                K_Lm = np.ma.masked_array(K_L, M_L)
                D_L = 1 - K_Lm

                sem_novelty_amean = D_L.mean(axis=1)
                sem_novelty_max = D_L.max(axis=1)
                sem_novelty_gmean = gmean(D_L, axis=1)

                df["SUM_sem_novelty_amean"] = sem_novelty_amean
                df["SUM_sem_novelty_max"] = sem_novelty_max
                df["SUM_sem_novelty_gmean"] = sem_novelty_gmean




                K = (K > 0).astype("int32")
                degrees = K.sum(axis=1) - 1
                edges_x_2 = K.sum() - K.shape[0]
                if edges_x_2 == 0: edges_x_2 = 1
                pr = 1. - degrees / float(edges_x_2)
                df["SUM_pagerank"] = pr

                K_L = (K_L > .5).astype("int32")
                degrees_L = K_L.sum(axis=1) - 1
                edges_x_2_L = K_L.sum() - K.shape[0]
                if edges_x_2_L == 0: edges_x_2_L = 1
                pr_L = 1. - degrees_L / float(edges_x_2_L)
                df["SUM_sem_pagerank"] = pr_L


                print df["pretty text"]
               # print df[["SUM_sbasic_sum", "SUM_sbasic_amean", "SUM_sbasic_max"]]
               # print df[
               #     ["SUM_pagerank", "SUM_centrality", "SUM_novelty_gmean", 
               #      "SUM_novelty_amean", "SUM_novelty_max"]]

                ### Stream Features ###
                for key, val in doc_counts.items():
                    stream_uni_counts[key] += val      
                denom = stream_uni_counts["__TOTAL__"]
                sum_probs = []
                amean_probs = []
                max_probs = []
                
                for lemmas in df["lemmas stopped"].tolist():
                    probs = [stream_uni_counts[lem.lower()] / denom for lem in lemmas]
                    sum_probs.append(np.sum(probs))
                    amean_probs.append(np.mean(probs))
                    max_probs.append(np.max(probs))

                df["STREAM_sbasic_sum"] = sum_probs
                df["STREAM_sbasic_amean"] = amean_probs
                df["STREAM_sbasic_max"] = max_probs



                for lemmas, nes in izip(df["lemmas"].tolist(), df["ne"].tolist()):
                    for lem, ne in izip(lemmas, nes):
                        if ne == "PERSON":
                            stream_per_counts[lem.lower()] += 1
                            stream_per_counts["__TOTAL__"] += 1.                                    
                        if ne == "LOCATION":
                            stream_loc_counts[lem.lower()] += 1
                            stream_loc_counts["__TOTAL__"] += 1.                                    
                        if ne == "ORGANIZATION":
                            stream_org_counts[lem.lower()] += 1
                            stream_org_counts["__TOTAL__"] += 1.                                    


                for tuples in df["num tuples"].tolist():

                    for nt in tuples:
                        for item in nt:
                            stream_nt_counts[item.lower()] += 1
                            stream_nt_counts["__TOTAL__"] += 1.

                pdenom = stream_per_counts["__TOTAL__"]                
                ldenom = stream_loc_counts["__TOTAL__"]                
                odenom = stream_org_counts["__TOTAL__"]                
                ntdenom = stream_nt_counts["__TOTAL__"]                
                sum_per_probs = []
                amean_per_probs = []
                max_per_probs = []
                sum_loc_probs = []
                amean_loc_probs = []
                max_loc_probs = []
                sum_org_probs = []
                amean_org_probs = []
                max_org_probs = []
                sum_nt_probs = []
                amean_nt_probs = []
                max_nt_probs = []

                            
                for tuples in df["num tuples"].tolist():
                    if ntdenom > 0:
                        nt_probs = [stream_nt_counts[item.lower()] / ntdenom
                                    for nt in tuples
                                    for item in nt]    
                    else:
                        nt_probs = []

                    if len(nt_probs) > 0:
                        sum_nt_probs.append(np.sum(nt_probs))
                        amean_nt_probs.append(np.mean(nt_probs))
                        max_nt_probs.append(np.max(nt_probs))
                    else:
                        sum_nt_probs.append(0)
                        amean_nt_probs.append(0)
                        max_nt_probs.append(0)


                for lemmas, nes in izip(df["lemmas"].tolist(), df["ne"].tolist()):

                    

                    if pdenom > 0:
                        per_probs = [stream_per_counts[lem.lower()] / pdenom
                                     for lem, ne in izip(lemmas, nes)
                                     if ne == "PERSON"]
                    else:
                        per_probs = []

                    if len(per_probs) > 0:
                        sum_per_probs.append(np.sum(per_probs))
                        amean_per_probs.append(np.mean(per_probs))
                        max_per_probs.append(np.max(per_probs))
                    else:
                        sum_per_probs.append(0)
                        amean_per_probs.append(0)
                        max_per_probs.append(0)

                    if ldenom > 0:
                        loc_probs = [stream_loc_counts[lem.lower()] / ldenom
                                     for lem, ne in izip(lemmas, nes)
                                     if ne == "LOCATION"]
                    else:
                        loc_probs = []

                    if len(loc_probs) > 0 :
                        sum_loc_probs.append(np.sum(loc_probs))
                        amean_loc_probs.append(np.mean(loc_probs))
                        max_loc_probs.append(np.max(loc_probs))
                    else:
                        sum_loc_probs.append(0)
                        amean_loc_probs.append(0)
                        max_loc_probs.append(0)

                    if odenom > 0:
                        org_probs = [stream_org_counts[lem.lower()] / odenom
                                     for lem, ne in izip(lemmas, nes)
                                     if ne == "ORGANIZATION"]
                    else:
                        org_probs = []

                    if len(org_probs) > 0 :
                        sum_org_probs.append(np.sum(org_probs))
                        amean_org_probs.append(np.mean(org_probs))
                        max_org_probs.append(np.max(org_probs))
                    else:
                        sum_org_probs.append(0)
                        amean_org_probs.append(0)
                        max_org_probs.append(0)



                df["STREAM_per_prob_sum"] = sum_per_probs
                df["STREAM_per_prob_max"] = max_per_probs
                df["STREAM_per_prob_amean"] = amean_per_probs

                df["STREAM_loc_prob_sum"] = sum_loc_probs
                df["STREAM_loc_prob_max"] = max_loc_probs
                df["STREAM_loc_prob_amean"] = amean_loc_probs

                df["STREAM_org_prob_sum"] = sum_org_probs
                df["STREAM_org_prob_max"] = max_org_probs
                df["STREAM_org_prob_amean"] = amean_org_probs

                df["STREAM_nt_prob_sum"] = sum_nt_probs
                df["STREAM_nt_prob_max"] = max_nt_probs
                df["STREAM_nt_prob_amean"] = amean_nt_probs

                #print df[["STREAM_sbasic_sum", "STREAM_sbasic_amean", "STREAM_sbasic_max"]]
                #print df[["STREAM_per_prob_sum", "STREAM_per_prob_amean", "STREAM_per_prob_max"]]  
                #print df[["STREAM_loc_prob_sum", "STREAM_loc_prob_amean", "STREAM_loc_prob_max"]]  
                #print df[["STREAM_nt_prob_sum", "STREAM_nt_prob_amean", "STREAM_nt_prob_max"]]  


                ### Write dataframe to file ###
                df[all_cols].to_csv(f, index=False, header=False, sep="\t")
                

                #print df[all_cols]

#            doc_persons = set()
#            for sent in doc:
#                for ne, ne_type, span in sent.get_ne_spans():
#                    if ne_type == u"PERSON":
#                        mention_counts[ne] += 1
#                        total_mentions += 1
#                        doc_persons.add(ne)
#            dep_feats = map(get_number_feats, doc)
#            print dep_feats
#            for per in doc_persons:
#                print per, mention_counts[per], mention_counts[per] / float(total_mentions)
#            assert len(doc) == len(df)
#            print "---"
            #print df[["lm", "sent text"]]
            #print 

    def compute_query_features(self, df, 
            query, synonyms, hypernyms, hyponyms):

        query_sent_covs = []
        sent_query_covs = []
        syn_sent_covs = []
        sent_syn_covs = []
        hyper_sent_covs = []
        sent_hyper_covs = []
        hypo_sent_covs = []
        sent_hypo_covs = []

        for tokens, lemmas in izip(
                df["tokens"].tolist(), df["lemmas"].tolist()):

            query_sent_cov = 0
            sent_query_cov = 0
            syn_sent_cov = 0
            sent_syn_cov = 0
            hyper_sent_cov = 0
            sent_hyper_cov = 0
            hypo_sent_cov = 0
            sent_hypo_cov = 0

            tokens = set([t.lower() for t in tokens])
            lemmas = set(lemmas)
            for q in query:
                if q in tokens or q in lemmas:
                    query_sent_cov += 1
                    sent_query_cov += 1

            sent_query_cov /= float(len(query))
            sent_query_covs.append(sent_query_cov)
            query_sent_cov /= float(len(tokens))
            query_sent_covs.append(query_sent_cov)
        
            for s in synonyms:
                if s in tokens or s in lemmas:
                    syn_sent_cov += 1
                    sent_syn_cov += 1
            sent_syn_cov /= float(len(synonyms))
            sent_syn_covs.append(sent_syn_cov)
            syn_sent_cov /= float(len(tokens))
            syn_sent_covs.append(syn_sent_cov)
 
            for s in hyponyms:
                if s in tokens or s in lemmas:
                    hypo_sent_cov += 1
                    sent_hypo_cov += 1
            sent_hypo_cov /= float(len(hyponyms))
            sent_hypo_covs.append(sent_hypo_cov)
            hypo_sent_cov /= float(len(tokens))
            hypo_sent_covs.append(hypo_sent_cov)
 
            for s in hypernyms:
                if s in tokens or s in lemmas:
                    hyper_sent_cov += 1
                    sent_hyper_cov += 1
            sent_hyper_cov /= float(len(hypernyms))
            sent_hyper_covs.append(sent_hyper_cov)
            hyper_sent_cov /= float(len(tokens))
            hyper_sent_covs.append(hyper_sent_cov)
 
        df["Q_query_sent_cov"] = query_sent_covs
        df["Q_sent_query_cov"] = sent_query_covs

        df["Q_syn_sent_cov"] = syn_sent_covs
        df["Q_sent_syn_cov"] = sent_syn_covs

        df["Q_hypo_sent_cov"] = hypo_sent_covs
        df["Q_sent_hypo_cov"] = sent_hypo_covs

        df["Q_hyper_sent_cov"] = hyper_sent_covs
        df["Q_sent_hyper_cov"] = sent_hyper_covs
