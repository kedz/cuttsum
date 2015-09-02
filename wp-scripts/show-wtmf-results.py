import pandas as pd

#natural-disasters.tsv
files = ["wp-wtmf-results/accidents.tsv", "wp-wtmf-results/terrorism.tsv",
         "wp-wtmf-results/social-unrest.tsv",
         "wp-wtmf-results/natural-disasters.tsv"
]

for path in files:
    with open(path, "r") as f:
        df = pd.read_csv(f, sep="\t")
    df = df.groupby("model").mean().sort("normP@100")
    df = df.reset_index()
    preprocs = set(df["model"].apply(lambda x: x.split(".lam_")[0]).tolist())

    data = []
    for preproc in preprocs:
        df_p = df[df["model"].apply(lambda x: preproc in x)]
        max_scores = df_p["normP@100"].max()
        df_p = df_p.sort("model")
        data.append((max_scores, preproc, df_p))

    data.sort(key=lambda x: x[0])

    for score, preproc, df_p in data:
        print score, preproc
        print df_p
        print
