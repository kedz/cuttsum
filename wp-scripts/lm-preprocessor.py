import corenlp as cnlp
import sys
import os
import gzip
from collections import defaultdict

def main(input_dir, output_path, norm, stop, ne, port):
    dirname, fname = os.path.split(output_path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

    output_path = os.path.join(
        dirname,
        "{}.norm-{}{}{}.spl.gz".format(
            fname, norm, ".stop" if stop else "", ".ne" if ne else ""))
    print "Writing spl file to {} ...".format(output_path)

    if not os.path.exists(input_dir):
        raise Exception("{} does not exist!".format(input_dir))
    paths = [os.path.join(input_dir, fname)
             for fname in os.listdir(input_dir)]

    texts = []
    for path in paths:
        with open(path, "r") as f:
            text = f.read()
        if text.strip() == "": continue
        texts.append(text)
    print "Removed", len(paths) - len(texts), "empty files."

    if ne is True:
        annotators = ["tokenize", "ssplit", "pos", "lemma", "ner"]
    elif norm == "lemma":
        annotators = ["tokenize", "ssplit", "pos", "lemma"]
    else:
        annotators = ["tokenize", "ssplit"]

    if norm == "stem":
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()

    if stop:
        with open("stopwords.txt", "r") as f:
            sw = set([word.strip().decode("utf-8").lower() for word in f])
    

    with cnlp.Server(annotators=annotators, mem="6G",
            port=port,
            max_message_len=1000000) as client, \
            gzip.open(output_path, "w") as f:
        
        for i, text in enumerate(texts):
            sys.stdout.write("{:3.1f}%\r".format(i * 100. / len(texts)))
            sys.stdout.flush()
            doc = client.annotate(text)

            if ne:
                per_counts = defaultdict(int)
                org_counts = defaultdict(int)
                loc_counts = defaultdict(int)
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

            for sent in doc:
                if ne:
                    
                    toks = []
                    for tok in sent:
                        if tok.ne == "PERSON":
                            if unicode(tok.lem).lower() == central_per:
                                print tok, "__CPER__"
                                toks.append(u"__CPER__")
                            else:
                                toks.append(u"__PER__")
                        elif tok.ne == "LOCATION":
                            if unicode(tok.lem).lower() == central_loc:
                                print tok, "__CLOC__"
                                toks.append(u"__CLOC__")
                            else:
                                toks.append(u"__LOC__")

                        elif tok.ne == "ORGANIZATION":
                            if unicode(tok.lem).lower() == central_org:
                                print tok, "__CORG__"
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
                        toks = [unicode(tok.lem).lower() for tok in sent]
                    elif norm == "stem":
                        toks = [stemmer.stem(unicode(tok).lower())
                                for tok in sent]
                    else:
                        toks = [unicode(tok).lower() for tok in sent]
                    if stop:
                        toks = [tok for tok in toks if tok not in sw]
                    toks = [tok for tok in toks if len(tok) < 50]
                if len(toks) == 0: continue
                string = u" ".join(toks).encode("utf-8") + "\n"
                print string
                f.write(string)



if __name__ == u"__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--norm", choices=["stem", "lemma", "none"], type=str, required=True)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--ne", action="store_true")
    parser.add_argument("--port", type=int, default=1986, required=False)
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output, args.norm, args.stop, args.ne, args.port)

