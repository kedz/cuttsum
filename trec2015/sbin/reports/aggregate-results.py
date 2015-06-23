import sys
import os
import cuttsum.events
import pandas as pd
pd.set_option('display.width', 200)



def split_path(path):
    splits = []
    while 1:
        path, base = os.path.split(path)
        if base == "":
            splits.append(path)
            break
        splits.append(base)
    return splits

def main(base_dir):

    events = cuttsum.events.get_events()
    test_data = []

    for path, dirs, filenames in os.walk(base_dir):
        print path
        splits = split_path(path)
        print splits
        if splits[0] == "test":
            print "HERE"
            n_iter = int(splits[1].split("-")[1])
            model = splits[2]
            query_num = int(splits[3].split("-")[1])
            event = [e for e in events if e.query_num == query_num][0]                

            print model, n_iter, query_num, event.fs_name()
            tsv_path = os.path.join(path, "{}.tsv".format(event.fs_name()))
            if os.path.exists(tsv_path):
                with open(tsv_path, "r") as f:
                    df = pd.read_csv(f, sep="\t")
                    inst = df.tail(1)[["avg. gain", "rec", "acc"]].to_dict(
                        orient="records")[0]
                    inst["model"] = model
                    inst["iter"] = n_iter
                    inst["event"] = event.fs_name()
                    test_data.append(inst)
        

        elif splits[0] == "train":
            pass
    test_df = pd.DataFrame(test_data)
    test_df = test_df[["model", "iter", "avg. gain", "rec", "acc"]].groupby(["model", "iter"]).mean()
    #test_df = test_df.sort()
    print test_df

if __name__ == u"__main__":

    if len(sys.argv) > 1:
        dirname = sys.argv[1]
    main(dirname)


