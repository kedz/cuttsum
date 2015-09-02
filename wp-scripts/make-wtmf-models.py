import os
import gzip
import wtmf
from sklearn.externals import joblib

def main(input_path, output_path, norm, stop, ne, lam):

    input_path = "{}.norm-{}{}{}.spl.gz".format(
        input_path, norm, ".stop" if stop else "", ".ne" if ne else "")

    dirname, fname = os.path.split(output_path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

    output_path = os.path.join(
        dirname,
        "{}.norm-{}{}{}.lam{:0.3f}.pkl".format(
            fname, norm, ".stop" if stop else "", ".ne" if ne else "", lam))

    with gzip.open(input_path, "r") as f:
        X_text = [line.strip() for line in f.readlines()]
    
    vec = wtmf.WTMFVectorizer(input='content', lam=lam, tf_threshold=2, verbose=True)
    vec.fit(X_text)
    print "Writing pkl file to {} ...".format(output_path)
    joblib.dump(vec, output_path)
    

if __name__ == u"__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--norm", choices=["stem", "lemma", "none"], type=str, required=True)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--ne", action="store_true")
    parser.add_argument("--lam", type=float, required=True)
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.norm, args.stop, args.ne, args.lam)
