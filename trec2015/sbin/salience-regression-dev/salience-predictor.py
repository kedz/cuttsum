import os
import cuttsum.events
import cuttsum.corpora
from cuttsum.pipeline import InputStreamResource
import numpy as np
import pandas as pd

def ds(X, y):
    I = np.arange(y.shape[0])
    I_1 = I[y[:,0] == 1]
    I_0 = I[y[:,0] == 0]
    size = I_1.shape[0]

    np.random.shuffle(I_0)
    I_0_ds = I_0[:size]

    X_0 = X[I_0_ds]
    X_1 = X[I_1]
    y_ds = np.zeros((size * 2,), dtype="int32")
    y_ds[size:2*size] = 1

    X_ds = np.vstack([X_0, X_1])
    I_ds = np.arange(size * 2)
    np.random.shuffle(I_ds)
    X_ds = X_ds[I_ds]
    y_ds = y_ds[I_ds]
    return X_ds, y_ds

def get_input_stream(event, extractor="goose", thresh=.8, delay=None, topk=20):
    corpus = cuttsum.corpora.get_raw_corpus(event)
    res = InputStreamResource()
    return res.get_dataframes(event, corpus, extractor, thresh, delay, topk)




dirname = "sp"
if not os.path.exists(dirname):
    os.makedirs(dirname)

basic_cols = ["BASIC length", "BASIC char length",
    "BASIC doc position", "BASIC all caps ratio",
    "BASIC upper ratio", 
    #"BASIC lower ratio",
    #"BASIC punc ratio", 
    "BASIC person ratio",
    "BASIC location ratio",
    "BASIC organization ratio", "BASIC date ratio",
    "BASIC time ratio", "BASIC duration ratio",
    "BASIC number ratio", "BASIC ordinal ratio",
    "BASIC percent ratio", "BASIC money ratio",
    "BASIC set ratio", "BASIC misc ratio"]

query_bw_cols = [
    "Q_sent_query_cov",
    "Q_sent_syn_cov",
    "Q_sent_hyper_cov",
    "Q_sent_hypo_cov",
]

query_fw_cols = [
    "Q_query_sent_cov",
    "Q_syn_sent_cov",
    "Q_hyper_sent_cov",
    "Q_hypo_sent_cov",
]

lm_cols = ["LM domain avg lp",
           "LM gw avg lp"]

sum_cols = [
    "SUM_sbasic_sum",
    "SUM_sbasic_amean",
#    "SUM_sbasic_max",
    "SUM_novelty_gmean",
    "SUM_novelty_amean",
#    "SUM_novelty_max",
    "SUM_centrality",
    "SUM_pagerank",
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


cols = basic_cols + query_fw_cols + lm_cols + sum_cols + stream_cols







events = [e for e in cuttsum.events.get_events()
          if e.query_num < 26 and e.query_num not in set([6, 7])]
events = events
y = []
X = []




istreams = []


for event in events:
    print event
    istream = get_input_stream(event)
    istreams.append(istream)
    y_e = []
    X_e = []
    for df in istream:

        selector = (df["n conf"] == 1) & (df["nugget probs"].apply(len) == 0)
        df.loc[selector, "nugget probs"] = df.loc[selector, "nuggets"].apply(lambda x: {n:1 for n in x})


        df["probs"] = df["nugget probs"].apply(lambda x: [val for key, val in x.items()] +[0])
        df["probs"] = df["probs"].apply(lambda x: np.max(x))
        df.loc[(df["n conf"] == 1) & (df["nuggets"].apply(len) == 0), "probs"] = 0
        y_t = df["probs"].values
        y_t = y_t[:, np.newaxis]
        y_e.append(y_t)
        X_t = df[cols].values
        X_e.append(X_t)

    y_e = np.vstack(y_e)
    y.append(y_e)
    X_e = np.vstack(X_e)
    X.append(X_e)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
num_events = len(events)

def hard_test(istream, cols):
    X = []
    y = []
    for df in istream:
        df_c = df[df["n conf"] == 1]
        y_t = (df_c["nuggets"].apply(len) > 0).values.astype("int32")
        y_t = y_t[:, np.newaxis]
        y.append(y_t)
        X_t = df_c[cols].values
        X.append(X_t)
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y


clfs = []
results = []
for i in xrange(num_events):
    X_test, y_test = hard_test(istreams[i], cols)
    
    X_test_s = X[i]
    y_test_s = y[i]
    X_train = np.vstack(X[0:i] + X[i+1:])
    y_train = np.vstack(y[0:i] + y[i+1:])
    
#    X_train, y_train = ds(X_train, y_train)

    for max_depth in [3]: #, 2, 3, 4, 5]:
        for lr in [1.,]: # 1.,]:
            for est in [100, ]:
                gbc = GradientBoostingRegressor(n_estimators=est, learning_rate=lr,
                    max_depth=max_depth, random_state=0)

                gbc.fit(X_train, y_train.ravel())
                y_pred = gbc.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                #prec, rec, fscore, sup = precision_recall_fscore_support(y_test, y_pred)
                I = y_test.ravel().argsort()
                item_index = np.where(y_test.ravel()[I] == 1)
                print y_test.ravel()[I][:item_index[0][0]]
                neg_mean = np.mean(y_pred.ravel()[I][:item_index[0][0]])
                pos_mean = np.mean(y_pred.ravel()[I][item_index[0][0]:])
                
                #print I
                plt.plot(y_pred.ravel()[I], label="predicted")
                #plt.plot(np.sort(y_test.ravel()), label="truth")
                plt.plot(y_test.ravel()[I], label="truth")
                plt.plot([0,item_index[0][0]], [neg_mean] * 2, "-.", label="neg mean")
                plt.plot([item_index[0][0],item_index[0][-1]], [pos_mean] * 2, "-.", label="neg mean")
                
                #plt.legend()
                plt.gcf().suptitle(events[i].fs_name())
                plt.savefig(os.path.join(dirname, "{}_hard.png".format(events[i].fs_name())))
                plt.close("all")

                
                y_pred = gbc.predict(X_test_s)
                mse = mean_squared_error(y_test_s, y_pred)
                I = y_test_s.ravel().argsort()
            
                #print I
                #print y_test
                plt.plot(y_pred.ravel()[I], label="predicted")
                #plt.plot(np.sort(y_test.ravel()), label="truth")
                plt.plot(y_test_s.ravel()[I], label="truth")
                #plt.legend()
                plt.gcf().suptitle(events[i].fs_name())
                plt.savefig(os.path.join(dirname, "{}_soft.png".format(events[i].fs_name())))
                plt.close("all")
                
                plt.hist(y_test_s.ravel() - y_pred.ravel(), bins=25)
                
                plt.gcf().suptitle(events[i].fs_name()+" residuals")
                plt.savefig(os.path.join(dirname, "{}_res.png".format(events[i].fs_name())))
                plt.close("all")
                
                results.append(
                    {"name": "gbc lr_{} est_{} dep_{}".format(lr, est, max_depth),
                     "event": events[i].fs_name(),
                     "mse": mse,
                     #"pos prec": prec[1],
                     #"pos recall": rec[1],
                     #"pos fscore": fscore[1],
                     #"pos support": sup[1],
                    })
                print results[-1]["name"]

df = pd.DataFrame(results)
#u_df = df[["name", "pos prec", "pos recall", "pos fscore"]].groupby("name").agg("mean")
u_df = df[["name", "mse"]].groupby("name").agg("mean")
print u_df
print
idx = u_df["mse"].argmax()
print u_df.loc[idx]

