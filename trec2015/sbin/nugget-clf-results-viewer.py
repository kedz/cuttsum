import pandas as pd
import os

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

def main():
    results_dir = os.path.join(os.getenv("TREC_DATA", "."), "nugget-classifiers")
    results = []
    for path, dirs, files in os.walk(results_dir):
        if len(files) > 0:
            event, nugget_id = path.split(os.sep)[-2:]
            with open(os.path.join(path, "stats.tsv"), "r") as f:
                df = pd.read_csv(f, sep="\t")
                df = df[df["coverage"] >= 0.5]
                df = df[df["pos prec"] >= 0.7]
                if len(df) == 0:
                    continue
                df.reset_index(inplace=True)
                argmax = df.iloc[df["pos prec"].argmax()]
                argmax["event title"] = event
                results.append(argmax)

    df = pd.DataFrame(results)
    df = df.set_index(["event title"])
    print df
                


if __name__ == u"__main__":
    main()
