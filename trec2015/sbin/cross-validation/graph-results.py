import pandas as pd
import os
import matplotlib.pylab as plt
plt.style.use('ggplot')


import sys

dirname = sys.argv[1]

def fmeasure(p, r):
    return 2 * (p * r) / (p + r)


with open(os.path.join(dirname, "scores.tsv"), "r") as f:
    df = pd.read_csv(f, sep="\t")

    mean_df = pd.concat([group.mean().to_frame().transpose()
                         for niter, group in df.groupby("iter")])
    mean_df["F1"] = fmeasure(mean_df["E[gain]"].values, mean_df["Comp."])
    print mean_df
    x = mean_df["iter"].values
    plt.close("all")
    plt.plot(x, mean_df["Comp."].values, "b", label="$\mathrm{Comp.}$")
    plt.plot(x, mean_df["E[gain]"].values, "g", label="$\mathbb{E}[\mathrm{gain}]$")
    plt.plot(x, mean_df["F1"].values, "r", label="$F_1$")

    plt.xlabel("iters")
    plt.ylabel("score")
    plt.xticks(range(1, 21))
    plt.gca().set_xlim([0.5, 20.5])
    plt.legend()
    plt.gcf().suptitle("Mean Scores")
    plt.savefig(os.path.join(dirname, "mean.scores.png"))

    plt.close("all")
    plt.plot(x, mean_df["Loss"].values, "y", label="$\mathrm{Loss}$")
    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.xticks(range(1, 21))
    plt.gca().set_xlim([0.5, 20.5])
    plt.gcf().suptitle("Mean Loss")
    plt.savefig(os.path.join(dirname, "mean.loss.png"))



for qid, event_scores in df.groupby("event"):
    x = event_scores["iter"].values
    plt.close("all")
    f = fmeasure(
        event_scores["E[gain]"].values,    
        event_scores["Comp."].values)

    plt.plot(x, event_scores["Comp."], "b", label="$\mathrm{Comp.}$")
    plt.plot(mean_df["iter"].values, mean_df["Comp."].values, "b--", alpha=.2)
    plt.plot(x, event_scores["E[gain]"], "g", label="$\mathbb{E}[\mathrm{gain}]$")
    plt.plot(mean_df["iter"].values, mean_df["E[gain]"].values, "g--", alpha=.2)
    plt.plot(x, f, "r", label="$F_1$")
    plt.plot(x, mean_df["F1"].values, "r--", alpha=.2)

    plt.xlabel("iters")
    plt.ylabel("score")
    plt.xticks(range(1, 21))
    plt.gca().set_xlim([0.5, 20.5])
    plt.legend()
    plt.gcf().suptitle("{} Scores".format(qid))
    plt.savefig(os.path.join(dirname, "{}.scores.png".format(qid)))

    plt.close("all")
    plt.plot(x, event_scores["Loss"], "y", label="$\mathrm{Loss}$")
    plt.plot(mean_df["iter"].values, mean_df["Loss"].values, "y--", alpha=.2)
    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.xticks(range(1, 21))
    plt.gca().set_xlim([0.5, 20.5])
    #plt.legend()
    plt.gcf().suptitle("{} Loss".format(qid))
    print qid
    print os.path.join(dirname, "{}.loss.png".format(qid))
    plt.savefig(os.path.join(dirname, "{}.loss.png".format(qid)))


with open(os.path.join(dirname,"weights.tsv"), "r") as f:

    df = pd.read_csv(f, sep="\t")

events = set(df["event"].tolist())

for clazz in ["SELECT", "NEXT"]:
    for event in events:
        for niter in xrange(1, 21):

            b = (df["event"] == event) & (df["class"] == clazz) & (df["iter"] == niter)  
            df.loc[b, "rank"] = df.loc[b]["weight"].argsort()


for name, fweights in df.groupby("name"):
    #print df.loc[df["name"] == name]
    plt.close("all")
    print name
    for clazz, avg_weights in fweights.groupby(["iter", "class"]).mean().reset_index().groupby("class"):
        if clazz == "SELECT":
            plt.plot(avg_weights["iter"].values, avg_weights["weight"].values, "g", label="$select$")
            for x, y, rank in zip(avg_weights["iter"].values, avg_weights["weight"].values, avg_weights["rank"].values):
                plt.gca().text(x, y + .0001, "$r_\mu={:0.1f}$".format(rank), fontsize=6)

        else:
            plt.plot(avg_weights["iter"].values, avg_weights["weight"].values, "r", label="$next$")
            for x, y, rank in zip(avg_weights["iter"].values, avg_weights["weight"].values, avg_weights["rank"].values):
                plt.gca().text(x, y+.0001, "$r_\mu={:0.1f}$".format(rank), fontsize=6)

    for qid, fweights_by_qid in fweights.groupby("event"):
        for clazz, fq_by_c_qid in fweights_by_qid.groupby("class"):
            #print qid, clazz, fq_by_c_qid
            if clazz == "SELECT":
                plt.plot(fq_by_c_qid["iter"].values, fq_by_c_qid["weight"].values, "g", alpha=.1)
            else:
                plt.plot(fq_by_c_qid["iter"].values, fq_by_c_qid["weight"].values, "r", alpha=.1)
    plt.xticks(range(1,21))
    plt.xlabel("iter")
    plt.ylabel("weight")
    plt.legend()
    plt.gca().set_xlim([0.5, 20.5])
    plt.gcf().suptitle("feature {}".format(name.replace(" ", "_")))
    plt.savefig(os.path.join(dirname, "feat.{}.png".format(name.replace(" ", "_"))))





