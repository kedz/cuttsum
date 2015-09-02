import wtmf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.externals import joblib
import corenlp as cnlp
import cuttsum.judgements
import gzip

matches_df = cuttsum.judgements.get_merged_dataframe()
nuggets_df = cuttsum.judgements.get_nuggets()

annotators = ["tokenize", "ssplit"]
with cnlp.Server(annotators=annotators, mem="6G",
        max_message_len=1000000) as client:

 

    for event, event_nuggets in nuggets_df.groupby('query id'):
        nugget_docs = []
        for text in event_nuggets["text"].tolist():
            doc = client.annotate(text)
            print doc
            nugget_docs.append(nugget_docs)
#    for nugget in nuggets_df.iterrows():
#    print nugget

exit()

with gzip.open("wp-lm-preproc/accidents.norm-lemma.stop.spl.gz", "r") as f:
    X = f.readlines()

Xtxt = X
print len(X)

vec = wtmf.WTMFVectorizer(input='content', lam=20., verbose=True, tf_threshold=2)
vec.fit(Xtxt)
joblib.dump(vec, "tmp.pkl")
vec = joblib.load("tmp.pkl")



X = vec.transform(Xtxt)

K = cosine_similarity(X[:20])

for i, k in enumerate(K):
    print i, Xtxt[i]
    for j in k.argsort()[::-1][:5]:
        print K[i,j], Xtxt[j]
    print

    

exit()

with open("version-14-10-17/mymodel/train.ind", "r") as f:
    max_row = 0
    max_col = 0
    data = []
    for line in f:
        c, r, v = line.strip().split("\t")
        r = int(r) 
        c = int(c) 
        v = float(v)
        max_row = max(r, max_row)
        max_col = max(c, max_col)
        print r,c,v
        data.append((r-1,c-1,v))
    Xsp = np.zeros(shape=(max_row, max_col), dtype="float64")
    print Xsp.shape
    for r, c, v in data:
        Xsp[r, c] = v

    print Xsp

    trans = wtmf.WTMFTransformer(lam=20., verbose=True)
    trans.fit(Xsp)
    print trans.P_

exit()

vec = wtmf.WTMFVectorizer(input='content', verbose=True)
with open("wp-sl/accident.txt", "r") as f:
    for i in xrange(103000):
        f.readline()
    X_t = [line.strip() for line in f.readlines()]
vec.fit(X_t)
X = vec.transform(X_t)

K = cosine_similarity(X)

for i, row in enumerate(K):
    print i, X_t[i]
    for j in K[i].argsort()[-5:]:
        print K[i,j], X_t[j]
    print "\n"


