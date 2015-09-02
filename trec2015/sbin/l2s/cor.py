import pandas as pd

with open("ssize.100.spe.10.l2.0.iters.5.fltr/summaries.tsv", "r") as f:
    df = pd.read_csv(f, sep="\t")
    df = df[df["run"] == "L2S.F1"]
print df[["partial", "confidence"]].corr("pearson")
print df[["partial", "confidence"]].corr("spearman")
print df[["partial", "confidence"]].corr("kendall")
