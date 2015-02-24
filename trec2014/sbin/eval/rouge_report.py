rank = '/local/nlp/kedzie/trec2015/data/rouge/rank_sal_1.8_sim_0.4_result'
apsaltr = '/local/nlp/kedzie/trec2015/data/rouge/apsaltr_sal_0.6_sim_0.6_result'
apsal = '/local/nlp/kedzie/trec2015/data/rouge/apsal_sal_0.4_sim_0.7_result'
ap = '/local/nlp/kedzie/trec2015/data/rouge/ap_sim_0.7_result'
hac = '/local/nlp/kedzie/trec2015/data/rouge/hac_dist1.35_sim_0.7_result'

import re
import math
import pandas as pd
import numpy as np

def load_data(path):
    unigram_data = []
    bigram_data = []
    with open(path, 'r') as f:
        for line in f:
            #X ROUGE-1 Eval 1.X R:0.32000 P:0.35644 F:0.33724
            line = line.strip()
            m = re.search("X ROUGE-(\d+) Eval \d+.X R:([^ ]+) P:([^ ]+) F:([^ ]+)", line)
            if m is not None:
                ngram, recall, prec, f1 = m.groups(1)
                recall = float(recall)
                prec = float(prec)
                f1 = float(f1)
                assert not math.isnan(prec)
                assert not math.isnan(recall)
                assert not math.isnan(f1)
                assert ngram == '1' or ngram == '2'
                if ngram == '1':
                    unigram_data.append([recall, prec, f1])
                elif ngram == '2':
                    bigram_data.append([recall, prec, f1])
    ug_df = pd.DataFrame(
        data=unigram_data, columns=['Unigram Recall', 'Unigram Prec.', 'Unigram F1'])
    assert pd.isnull(ug_df).any().any() == False #.all()

    bg_df = pd.DataFrame(
        bigram_data, columns=['Bigram Recall', 'Bigram Prec.', 'Bigram F1'])
    assert pd.isnull(bg_df).any().any() == False #.all()

    assert len(ug_df) == len(bg_df)
    df = ug_df.join((bg_df))
    assert pd.isnull(df).any().any() == False #.all()
    return df, pd.DataFrame(df.mean()).transpose()

def comp_sig(df1, df2, col):
    X1 = df1[col].as_matrix().astype(np.float64)
    X1 = X1[:, np.newaxis]

    X2 = df2[col].as_matrix().astype(np.float64)
    X2 = X2[:, np.newaxis]
    mu_thr = np.fabs(X1.mean() - X2.mean())
    X = np.hstack((X1, X2))
    r = 0
    n_iters = 1000
    for n_iter in xrange(n_iters):
        for i in xrange(X.shape[0]):
            np.random.shuffle(X[i,:])
        Xmu = X.mean(axis=0)
        if np.fabs(Xmu[0] - Xmu[1]) >= mu_thr:
            r += 1
    return float(r + 1) / (n_iters + 1)

def comp_sig(df1, df2, col):
    from scipy.stats import wilcoxon
    X1 = df1[col].as_matrix().astype(np.float64)
    X1 = np.array([x.mean() for x in np.split(X1, 21)])
    #X1 = X1[:, np.newaxis]

    X2 = df2[col].as_matrix().astype(np.float64)
    X2 = np.array([x.mean() for x in np.split(X2, 21)])
    #X2 = X2[:, np.newaxis]
    T, pval = wilcoxon(X1, X2, zero_method='wilcox')
    return T, pval

apsal_df, apsal_mu_df = load_data(apsal)
apsaltr_df, apsaltr_mu_df = load_data(apsaltr)
ap_df, ap_mu_df = load_data(ap)
rank_df, rank_mu_df = load_data(rank)
hac_df, hac_mu_df = load_data(hac)

avg_df = pd.concat([apsal_mu_df, apsaltr_mu_df, 
                    ap_mu_df, rank_mu_df, hac_mu_df])
avg_df.index = ['apsal', 'apsal tr', 'ap', 'rank', 'hac']
pd.set_option('display.width', 200)
print avg_df

print "APSAL Time Ranked vs AP: Unigram F1 :", comp_sig(apsaltr_df, ap_df, 'Unigram F1')
print "APSAL Time Ranked vs AP: Bigram F1 :", comp_sig(apsaltr_df, ap_df, 'Bigram F1')
print "APSAL Time Ranked vs AP: Unigram Prec. :", comp_sig(apsaltr_df, ap_df, 'Unigram Prec.')
print "APSAL Time Ranked vs AP: Bigram Prec. :", comp_sig(apsaltr_df, ap_df, 'Bigram Prec.')
print "APSAL Time Ranked vs AP: Unigram Recall :", comp_sig(apsaltr_df, ap_df, 'Unigram Recall')
print "APSAL Time Ranked vs AP: Bigram Recall :", comp_sig(apsaltr_df, ap_df, 'Bigram Recall')
print 

print "APSAL Time Ranked vs Rank: Unigram F1 :", comp_sig(apsaltr_df, rank_df, 'Unigram F1')
print "APSAL Time Ranked vs Rank: Bigram F1 :", comp_sig(apsaltr_df, rank_df, 'Bigram F1')
print "APSAL Time Ranked vs Rank: Unigram Prec. :", comp_sig(apsaltr_df, rank_df, 'Unigram Prec.')
print "APSAL Time Ranked vs Rank: Bigram Prec. :", comp_sig(apsaltr_df, rank_df, 'Bigram Prec.')
print "APSAL Time Ranked vs Rank: Unigram Recall :", comp_sig(apsaltr_df, rank_df, 'Unigram Recall')
print "APSAL Time Ranked vs Rank: Bigram Recall :", comp_sig(apsaltr_df, rank_df, 'Bigram Recall')
print

print "APSAL Time Ranked vs HAC: Unigram F1 :", comp_sig(apsaltr_df, hac_df, 'Unigram F1')
print "APSAL Time Ranked vs HAC: Bigram F1 :", comp_sig(apsaltr_df, hac_df, 'Bigram F1')
print "APSAL Time Ranked vs HAC: Unigram Prec. :", comp_sig(apsaltr_df, hac_df, 'Unigram Prec.')
print "APSAL Time Ranked vs HAC: Bigram Prec. :", comp_sig(apsaltr_df, hac_df, 'Bigram Prec.')
print "APSAL Time Ranked vs HAC: Unigram Recall :", comp_sig(apsaltr_df, hac_df, 'Unigram Recall')
print "APSAL Time Ranked vs HAC: Bigram Recall :", comp_sig(apsaltr_df, hac_df, 'Bigram Recall')
print



print "APSAL vs APSAL Time Ranked: Unigram F1 :", comp_sig(apsal_df, apsaltr_df, 'Unigram F1')
print "APSAL vs APSAL Time Ranked: Bigram F1 :", comp_sig(apsal_df, apsaltr_df, 'Bigram F1')
print "APSAL vs APSAL Time Ranked: Unigram Prec. :", comp_sig(apsal_df, apsaltr_df, 'Unigram Prec.')
print "APSAL vs APSAL Time Ranked: Bigram Prec. :", comp_sig(apsal_df, apsaltr_df, 'Bigram Prec.')
print "APSAL vs APSAL Time Ranked: Unigram Recall :", comp_sig(apsal_df, apsaltr_df, 'Unigram Recall')
print "APSAL vs APSAL Time Ranked: Bigram Recall :", comp_sig(apsal_df, apsaltr_df, 'Bigram Recall')
print
print "APSAL vs AP: Unigram F1 :", comp_sig(apsal_df, ap_df, 'Unigram F1')
print "APSAL vs AP: Bigram F1 :", comp_sig(apsal_df, ap_df, 'Bigram F1')
print "APSAL vs AP: Unigram Prec. :", comp_sig(apsal_df, ap_df, 'Unigram Prec.')
print "APSAL vs AP: Bigram Prec. :", comp_sig(apsal_df, ap_df, 'Bigram Prec.')
print "APSAL vs AP: Unigram Recall :", comp_sig(apsal_df, ap_df, 'Unigram Recall')
print "APSAL vs AP: Bigram Recall :", comp_sig(apsal_df, ap_df, 'Bigram Recall')
print 
print "APSAL vs HAC: Unigram F1 :", comp_sig(apsal_df, hac_df, 'Unigram F1')
print "APSAL vs HAC: Bigram F1 :", comp_sig(apsal_df, hac_df, 'Bigram F1')
print "APSAL vs HAC: Unigram Prec. :", comp_sig(apsal_df, hac_df, 'Unigram Prec.')
print "APSAL vs HAC: Bigram Prec. :", comp_sig(apsal_df, hac_df, 'Bigram Prec.')
print "APSAL vs HAC: Unigram Recall :", comp_sig(apsal_df, hac_df, 'Unigram Recall')
print "APSAL vs HAC: Bigram Recall :", comp_sig(apsal_df, hac_df, 'Bigram Recall')
print 
print "APSAL  vs HAC: Unigram F1 :", comp_sig(apsal_df, hac_df, 'Unigram F1')
print "APSAL  vs HAC: Bigram F1 :", comp_sig(apsal_df, hac_df, 'Bigram F1')
print "APSAL  vs HAC: Unigram Prec. :", comp_sig(apsal_df, hac_df, 'Unigram Prec.')
print "APSAL  vs HAC: Bigram Prec. :", comp_sig(apsal_df, hac_df, 'Bigram Prec.')
print "APSAL  vs HAC: Unigram Recall :", comp_sig(apsal_df, hac_df, 'Unigram Recall')
print "APSAL  vs HAC: Bigram Recall :", comp_sig(apsal_df, hac_df, 'Bigram Recall')
print


