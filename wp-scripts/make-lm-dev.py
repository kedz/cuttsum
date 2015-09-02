import corenlp as cnlp
import re
import os
import gzip
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


def main(output_path, norm, stop):

    dirname, fname = os.path.split(output_path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

    output_path = os.path.join(
        dirname,
        "{}.norm-{}{}.spl.gz".format(
            fname, norm, ".stop" if stop else ""))

    print "Domain: {}".format(fname)
    print "Output Path: {}".format(output_path)
    events = [event for event in cuttsum.events.get_events()
              if event.type in dom2type[fname] and event.query_num < 26]

    ne = False
    #if ne is True:
   #     annotators = ["tokenize", "ssplit", "pos", "lemma", "ner"]
    if norm == "lemma":
        annotators = ["tokenize", "ssplit", "pos", "lemma"]
    else:
        annotators = ["tokenize", "ssplit"]



    with cnlp.Server(annotators=annotators, mem="6G",
            port=2001, max_message_len=1000000) as client, \
            gzip.open(output_path, "w") as f:

        query_ids = set([event.query_id for event in events])
        updates = matches_df[matches_df["query id"].apply(lambda x: x in query_ids)]
        texts = updates.drop_duplicates(subset='update id')["update text"].apply(heal_text).tolist()
    
        central_per = None
        central_loc = None
        central_org = None

        print "processing update text"
        docs = [client.annotate(text) for text in texts] 
        for doc in docs[:10]:
            print doc
        print "tokenizing"
        X_upd_txt = tokenize(docs, norm, stop, ne, 
            central_per=central_per, central_loc=central_loc, 
            central_org=central_org)
        print "writing"
        for line in X_upd_txt:
            f.write(line + "\n")
 


if __name__ == u"__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=False, default=None)
    parser.add_argument("--norm", choices=["stem", "lemma", "none"], type=str, required=True)
    parser.add_argument("--stop", action="store_true", default=False)
    
    args = parser.parse_args()
    
    main(args.output, args.norm, args.stop)
