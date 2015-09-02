import sys
import os
import pandas as pd

path = os.path.join(sys.argv[1], "scores.tsv")
with open(path, "r") as f:
    df = pd.read_csv(f, sep="\t")

print df 
f1_idx = df["F1"].argmax()
comp_idx = df["Comp."].argmax()
egain_idx = df["E[gain]"].argmax()
loss_idx = df["Loss"].argmin()

df = pd.concat([
    df.iloc[f1_idx].to_frame().T,
    df.iloc[comp_idx].to_frame().T,
    df.iloc[egain_idx].to_frame().T,
    df.iloc[loss_idx].to_frame().T,
])

df.index = ["F1", "Comp.", "E[gain]", "Loss"]
df["iter"] = [f1_idx, comp_idx, egain_idx, loss_idx]
print df
