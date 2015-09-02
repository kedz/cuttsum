import os
import pandas as pd
import math
def sigmoid(x):
    return 1. / (1. + math.exp(-x))

if not os.path.exists("submissions"):
    os.makedirs("submissions")


cols = ["event", "team", "run", "stream id", "sentence id", "timestamp", "confidence", "text"]
with open(os.path.join("matched.ssize.100.spe.10.l2.0.iters.20", "summaries.tsv"), "r") as f:
    df = pd.read_csv(f, sep="\t")
df = df.loc[df["run"] == "L2S.F1"]
df["team"] = "cunlp"
df["run"] = "2LtoSnofltr20"
df["confidence"] = df["confidence"].apply(sigmoid)
df = df.loc[:, cols]
print len(df) / 46.

with open(os.path.join("submissions", "cunlp_2_l2s_nofltr_top20.tsv"), "w") as f:
    df[cols[:-1]].to_csv(f, sep="\t", index=False, header=False)
with open(os.path.join("submissions", "cunlp_2_l2s_nofltr_top20.sum.tsv"), "w") as f:
    df.to_csv(f, sep="\t", index=False, header=False)



cols = ["event", "org", "run", "stream id", "sent id", "timestamp", "conf", "part"]
with open(os.path.join("matched.ssize.100.spe.10.l2.0.iters.20.fltr", "submission.tsv"), "r") as f:
    df = pd.read_csv(f, sep="\t", header=None)
df.columns = cols
df = df.loc[df["run"] == "L2S.F1"]
df["org"] = "cunlp"
df["run"] = "1LtoSfltr20"
df["conf"] = df["conf"].apply(sigmoid)
df = df.loc[:, cols[:-1]]
print len(df) / 46.

with open(os.path.join("submissions", "cunlp_1_l2s_fltr_top20.tsv"), "w") as f:
    df.to_csv(f, sep="\t", index=False, header=False)

cols = ["event", "org", "run", "stream id", "sent id", "timestamp", "conf", "part"]
with open(os.path.join("matched.ssize.100.spe.10.l2.0.iters.20.fltr.trunc", "submission.tsv"), "r") as f:
    df = pd.read_csv(f, sep="\t", header=None)
df.columns = cols
df = df.loc[df["run"] == "L2S.F1"]
df["org"] = "cunlp"
df["run"] = "3LtoSfltr5"
df["conf"] = df["conf"].apply(sigmoid)
df = df.loc[:, cols[:-1]]
print len(df) / 46.

with open(os.path.join("submissions", "cunlp_3_l2s_fltr_top5.tsv"), "w") as f:
    df.to_csv(f, sep="\t", index=False, header=False)




#matched.ssize.100.spe.10.l2.0.iters.20.fltr    
#matched.ssize.100.spe.10.l2.0.iters.20.fltr.trunc 
