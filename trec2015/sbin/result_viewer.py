import pandas as pd
import os

pd.set_option('display.width', 200)

def main():
    results_dir = os.path.join(os.getenv("TREC_DATA", "."), "system-results")
    results = []
    for path, dirs, files in os.walk(results_dir):
        if len(files) > 0:
            for fname in files:
                if fname.endswith("stats.tsv"):
                    system, extractor, budget, soft_match, corpus =  path.split(os.sep)[-5:]
                    system = system.replace("-summaries", "")
                    system = system.replace("monotone-submodular", "mono-submod")
                    system = system.replace("retrospective", "retro")
                    filepath = os.path.join(path, fname)
                    with open(filepath, "r") as f:
                        df = pd.read_csv(f, sep="\t")
                        df.insert(3, "avg nz F(S)", df[df["F(S)"] > 0]["F(S)"].mean())
                        df = df.tail(1)
                        df["system"] = [system,]
                        df["extractor"] = [extractor,]
                        df["soft match"] = [soft_match == "soft_match"]
                        del df["F(S)"]
                        del df["timestamp"]
                        results.append(df)
    df = pd.concat(results)
    print df.set_index(["system", "extractor"])

if __name__ == u"__main__":
    main()
