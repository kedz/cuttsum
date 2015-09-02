import os
import gzip
from collections import defaultdict

def main(input_path, output_path, norm, stop, threshold):

    input_path = "{}.norm-{}{}.spl.gz".format(
        input_path, norm, ".stop" if stop else "")

    dirname, fname = os.path.split(output_path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

    output_path = os.path.join(
        dirname,
        "{}.norm-{}{}.spl.unk.gz".format(
            fname, norm, ".stop" if stop else ""))

    counts = defaultdict(int)
    with gzip.open(input_path, "r") as f:
        for line in f:
            for tok in line.strip().split(" "):
                counts[tok] += 1

    with gzip.open(input_path, "r") as f, gzip.open(output_path, "w") as o:
        for line in f:
            toks = [tok if counts[tok] >= threshold else "<unk>"
                    for tok in line.strip().split(" ")]
            o.write(' '.join(toks) + '\n')                        

if __name__ == u"__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--norm", choices=["stem", "lemma", "none"], type=str, required=True)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    main(args.input, args.output, args.norm, args.stop, args.threshold)
