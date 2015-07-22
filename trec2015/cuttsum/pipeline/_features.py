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
            return pd.read_csv(f, sep="\t", engine="python")


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


        synonyms, hypernyms, hyponyms = make_query_synsets()

        path = self.get_path(
            event, corpus, extractor, thresh)
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname): os.makedirs(dirname)

        meta_cols = ["update id", "stream id", "sent id", "timestamp", 
            "pretty text", "tokens", "lemmas", "pos", "ne", 
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
 
        all_cols = meta_cols + basic_cols + query_cols + lm_cols

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
                        " ".join([xi.lower() for xi in x])))
                dm_log_probs = [lp for lp, avg_lp in dm_probs.tolist()]
                dm_avg_log_probs = [avg_lp for lp, avg_lp in dm_probs.tolist()]
                df["LM domain lp"] = dm_log_probs
                df["LM domain avg lp"] = dm_avg_log_probs
                gw_probs = df["lemmas"].apply(
                    lambda x: gw_lm.sentence_log_prob(
                        " ".join([xi.lower() for xi in x])))
                gw_log_probs = [lp for lp, avg_lp in gw_probs.tolist()]
                gw_avg_log_probs = [avg_lp for lp, avg_lp in gw_probs.tolist()]
                df["LM gw lp"] = gw_log_probs
                df["LM gw avg lp"] = gw_avg_log_probs


                ### Query Features ###

                self.compute_query_features(df, 
                    set([q.lower() for q in event.query]),
                    synonyms, hypernyms, hyponyms)

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
