import pandas as pd
import cuttsum.events
from cuttsum.misc import event2lm_name

id2lm = {}
for event in cuttsum.events.get_events():
    if event.query_num == 7 or event.query_num > 25: continue
    id2lm[event.query_id] = event2lm_name(event)


with open("filters.tsv", "r") as f:
    df = pd.read_csv(f, sep="\t")

labels, bins = pd.cut(df["size"], bins=3, retbins=True)
df["bin"] = labels
#pd.cut( )

by_bin = []
for bin_, bin_df in df.groupby("bin"):
    print bin_
    bin_results = []

    for thr, results in bin_df.groupby("thr"):
        df_u = results.mean().to_frame().T 
        df_u["thr"] = thr
        bin_results.append(df_u)

    bin_results = pd.concat(bin_results)
    print bin_results

    idx = bin_results.reset_index(drop=True)["F1"].argmax()
    thr = bin_results.reset_index(drop=True).iloc[idx]["thr"]
    
    by_bin.append(bin_df[bin_df["thr"] == thr])

 
print pd.concat(by_bin).mean().to_frame().T


by_cat = []

df["thr"] = df["thr"].apply(lambda x: "{:0.2f}".format(x))
df["cat"] = df["event"].apply(lambda x: id2lm[x])
for cat, cat_df in df.groupby("cat"):
    print cat,
    cat_results = []

    for thr, results in cat_df.groupby("thr"):
        df_u = results.mean().to_frame().T 
        df_u["thr"] = thr
        cat_results.append(df_u)

    cat_results = pd.concat(cat_results)


    idx = cat_results.reset_index(drop=True)["F1"].argmax()
    thr = cat_results.reset_index(drop=True).iloc[idx]["thr"]
    print thr    
    by_cat.append(cat_df[cat_df["thr"] == thr])
    
    #by_cat.append(cat_results.reset_index(drop=True).iloc[idx].to_frame().T)

    #print idx
    #print

print pd.concat(by_cat).mean().to_frame().T


mean_results = []
for thr, results in df.groupby("thr"):
    tmp = results.mean().to_frame().T 
    tmp["thr"] = thr
    mean_results.append(tmp)
results = pd.concat(mean_results)
results.reset_index(drop=True, inplace=True)
print results.loc[results["F1"].argmax()].to_frame().T
