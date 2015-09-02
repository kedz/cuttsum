import corenlp as cnlp
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import gzip
import wtmf
from sklearn.externals import joblib
import cuttsum.events
import cuttsum.judgements
import pandas as pd
from collections import defaultdict

matches_df = cuttsum.judgements.get_merged_dataframe()

def heal_text(sent_text):
    sent_text = sent_text.decode("utf-8")
    sent_text = re.sub(
        ur"[a-z ]+, [a-z][a-z ]+\( [a-z]+ \) [-\u2014_]+ ", 
        r"", sent_text)
    sent_text = re.sub(
        ur"^.*?[a-z ]+, [a-z][a-z]+ [-\u2014_]+ ", 
        r"", sent_text)
    sent_text = re.sub(
        ur"^.*?[a-z ]+\([^\)]+\) [-\u2014_]+ ", 
        r"", sent_text)
    sent_text = re.sub(
        ur"^.*?[a-z]+ +[-\u2014_]+ ", 
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
    sent_text = re.sub(r"u\. (s|n)\.", r"u.\1.", sent_text)

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
        r"^photo:reuters|^photo:ap", 
        r"", sent_text)
    sent_text = sent_text.replace("\n", " ")

    return sent_text.encode("utf-8")



nuggets = cuttsum.judgements.get_nuggets()

updates = pd.concat([
    cuttsum.judgements.get_2013_updates(), 
    cuttsum.judgements.get_2014_sampled_updates()
])
#updates["text"] = updates["text"].apply(heal_text)

dom2type = {
    "accidents": set(["accident"]),
    "natural-disasters": set(["earthquake", "storm", "impact event"]),
    "social-unrest": set(["protest", "riot"]),
    "terrorism": set(["shooting", "bombing", "conflict", "hostage"]),
}


def tokenize(docs, norm, stop, ne, central_per=None, central_loc=None, central_org=None):

    if stop:
        with open("stopwords.txt", "r") as f:
            sw = set([word.strip().decode("utf-8").lower() for word in f])

    if norm == "stem":
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()

    all_toks = []
    for doc in docs:
        toks = []
        for sent in doc:
            if ne:                
                for tok in sent:
                    if tok.ne == "PERSON":
                        if unicode(tok.lem).lower() == central_per:
                            toks.append(u"__CPER__")
                        else:
                            toks.append(u"__PER__")
                    elif tok.ne == "LOCATION":
                        if unicode(tok.lem).lower() == central_loc:
                            toks.append(u"__CLOC__")
                        else:
                            toks.append(u"__LOC__")

                    elif tok.ne == "ORGANIZATION":
                        if unicode(tok.lem).lower() == central_org:
                            toks.append(u"__CORG__")
                        else:
                            toks.append(u"__ORG__")
                    else:
                        if norm == "lemma":
                            form = unicode(tok.lem).lower()
                        elif norm == "stem":
                            form = stemmer.stem(unicode(tok).lower())
                        else:
                            form = unicode(tok).lower()
                        if stop:
                            if form not in sw and len(form) < 50:
                                toks.append(form)
                        else:
                            if len(form) < 50:
                                toks.append(form)
            else:
                if norm == "lemma":
                    stoks = [unicode(tok.lem).lower() for tok in sent]
                elif norm == "stem":
                    stoks = [stemmer.stem(unicode(tok).lower())
                             for tok in sent]
                else:
                    stoks = [unicode(tok).lower() for tok in sent]
                if stop:
                    toks.extend([tok for tok in stoks if tok not in sw])
                else:
                    toks.extend(stoks)
 
        toks = [tok for tok in toks if len(tok) < 50]
        #if len(toks) == 0: continue
        string = u" ".join(toks).encode("utf-8")
        #print string
        all_toks.append(string)
    return all_toks





def find_central_nes(docs):
    per_counts = defaultdict(int)
    org_counts = defaultdict(int)
    loc_counts = defaultdict(int)
    for doc in docs:
        for sent in doc:
            for tok in sent:
                if tok.ne == "PERSON":
                    per_counts[unicode(tok.lem).lower()] += 1
                elif tok.ne == "LOCATION":
                    loc_counts[unicode(tok.lem).lower()] += 1
                elif tok.ne == "ORGANIZATION":
                    org_counts[unicode(tok.lem).lower()] += 1
        
        if len(per_counts) > 0:
            central_per = max(per_counts.items(), key=lambda x:[1])[0]
        else:
            central_per = None
        if len(org_counts) > 0:
            central_org = max(org_counts.items(), key=lambda x:[1])[0]
        else:
            central_org = None
        if len(loc_counts) > 0:
            central_loc = max(loc_counts.items(), key=lambda x:[1])[0]
        else:
            central_loc = None
    return central_per, central_loc, central_org


def main(input_path, output_path, norm, stop, ne, lam, port):
    dirname, domain = os.path.split(input_path)
    input_path = os.path.join(
        dirname,
        "{}.norm-{}{}{}.lam{:0.3f}.pkl".format(
            domain, norm, ".stop" if stop else "", ".ne" if ne else "", lam))
    print "Domain: {}".format(domain)
    print "Model Path: {}".format(input_path)
    events = [event for event in cuttsum.events.get_events()
              if event.type in dom2type[domain] and event.query_num < 26 and event.query_num != 7]

    if ne is True:
        annotators = ["tokenize", "ssplit", "pos", "lemma", "ner"]
    elif norm == "lemma":
        annotators = ["tokenize", "ssplit", "pos", "lemma"]
    else:
        annotators = ["tokenize", "ssplit"]


    results = []
    vec = joblib.load(input_path)

    modelname = "{}.norm_{}.stop_{}.ne_{}.lam_{}".format(domain, norm, stop, ne, lam)

    with cnlp.Server(annotators=annotators, mem="6G", port=port,
            max_message_len=1000000) as client:

        for event in events:
            print event
            event_nuggets = nuggets.loc[nuggets["query id"] == event.query_id]
            print "processing nugget text"
            nugget_docs = [client.annotate(text) 
                           for text in event_nuggets["text"].tolist()]
            #for doc in nugget_docs:
            #    print doc
            #print

            if ne:
                central_per, central_loc, central_org = find_central_nes(
                    nugget_docs)
            else:
                central_per = None
                central_loc = None
                central_org = None

            
            X_nug_txt = tokenize(nugget_docs, norm, stop, ne, 
                central_per=central_per, central_loc=central_loc, 
                central_org=central_org)
            nuggets.loc[nuggets["query id"] == event.query_id, "X"] = X_nug_txt
            event_nuggets = nuggets[nuggets["query id"] == event.query_id]
            event_nuggets = event_nuggets[event_nuggets["X"].apply(lambda x: len(x.split(" ")) < 50 and len(x.split(" ")) > 0)]
            X_nug_txt = event_nuggets["X"].tolist()
            #for txt in X_nug_txt:
            #    print txt
            #print
            print "transforming nugget text"
            X_nug = vec.transform(X_nug_txt)
            assert X_nug.shape[0] == len(event_nuggets)
            

            print "getting updates"
            updates.loc[updates["query id"] == event.query_id, "text"] = \
                updates.loc[updates["query id"] == event.query_id, "text"].apply(heal_text)

            event_updates = updates[(updates["query id"] == event.query_id) & (updates["text"].apply(len) < 1000)]
            print "processing update text"
            docs = [client.annotate(text) for text in event_updates["text"].tolist()] 
            X_upd_txt = tokenize(docs, norm, stop, ne, 
                central_per=central_per, central_loc=central_loc, 
                central_org=central_org)
            print "transforming update text"
            X_upd = vec.transform(X_upd_txt)
            
            for i, (index, nugget) in enumerate(event_nuggets.iterrows()):

                boolean = (matches_df["query id"] == event.query_id) & (matches_df["nugget id"] == nugget["nugget id"])
                match_ids = set(matches_df.loc[boolean, "update id"].tolist())
                if len(match_ids) == 0: continue

                #print index, nugget["nugget id"], nugget["text"]
                #print X_nug[i]
                if (X_nug[i] == 0).all(): continue

                n_matches = 0
                K = cosine_similarity(X_nug[i], X_upd)           
                for j in K.ravel().argsort()[::-1][:100]:
                    
                    #print K[0,j], 
                    #print event_updates.iloc[j]["text"]
                    if event_updates.iloc[j]["update id"] in match_ids:
                        n_matches += 1

                #print 
                P100 = n_matches / 100.
                optP100 = min(1., len(match_ids) / 100.)
                nP100 = P100 / optP100
                results.append(
                    {"model": modelname,
                     "nugget id": nugget["nugget id"], 
                     "P@100": P100, 
                     "opt P@100": optP100, 
                     "normP@100":nP100
                    })
            df = pd.DataFrame(results)
            print df
            print df["normP@100"].mean()
            df["model"] = modelname
        return results
#        print len(event_updates)

        #print event_updates["text"].apply(len).mean()
        #print event_updates["text"].apply(heal_text).apply(len).max()
        #print event_updates["text"].apply(heal_text).apply(len).median()
    
            
    

if __name__ == u"__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=False, default=None)
    parser.add_argument("--port", type=int, required=True)
#    parser.add_argument("--norm", choices=["stem", "lemma", "none"], type=str, required=True)
#    parser.add_argument("--stop", action="store_true")
#    parser.add_argument("--ne", action="store_true")
#    parser.add_argument("--lam", type=float, required=True)

    args = parser.parse_args()
    dirname = os.path.dirname(args.output)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

    data = []
    for norm in ["none", "lemma", "stem"]:
        for stop in [True, False]:
            for ne in [True, False]:
                for lam in [20., 10., 1., .1]:                     
                    data.extend(
                        main(args.input, args.output, norm, stop, ne, lam, args.port))

    df =  pd.DataFrame(data)
    with open(args.output, "w") as f:
        df.to_csv(f, sep="\t", index=False)

