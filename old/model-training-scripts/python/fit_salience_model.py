import os
import sys
import argparse
import numpy as np
from math import exp
import GPy
from sklearn.externals import joblib

def main():
    input_data, mode, model_dir, pred_file = parse_args()

    if mode == 'max':
        sim_index = 3
    elif mode == 'min':
        sim_index = 4
    else:
        sim_index = 5   

    y = []   
    X = []

    for sim_file, feat_file in input_data:
        sim_data = []
        sim_labels = []

        feat_data = []
        feat_labels = []
        print "Reading similarities from", sim_file
        with open(sim_file, 'r') as sf:
            sheader = sf.readline().strip().split('\t')
            for line in sf:
                datum = line.strip().split('\t')
                label = '{}-{}-{}'.format(datum[0], datum[1], datum[2])
                sim = float(datum[sim_index])
                sim_data.append(sim)
                sim_labels.append(label)                  
                
        print "Reading features from", feat_file
        with open(feat_file, 'r') as ff:
            fheader  = ff.readline().strip().split('\t')
            for line in ff:

                # 3: avg-tfidf 4: avg-tfidf-m1 5: avg-tfidf-m5 6: dm-logprob
                # 7: dm-avg-logprob 8: bg3-logprob 9: bg3-avg-logprob
                # 10: bg4-logprob 11: bg4-avg-logprob 12: bg5-logprob
                # 13: bg5-avg-logprob 14: query-matches 15: synset-matches 
                # 16: num-tokens 17: article-position 18: article-position-rel 
                # 19: capsrate 
                datum = line.strip().split('\t')
                label = '{}-{}-{}'.format(datum[0], datum[1], datum[2])
                feats = [float(datum[3]), float(datum[4]), float(datum[5]),
                         exp(float(datum[6])), exp(float(datum[7])),
                         exp(float(datum[12])), exp(float(datum[13])),
                         float(datum[14]), float(datum[15]), float(datum[16]),
                         float(datum[18]), float(datum[19])]
                         
                feat_data.append(feats)
                feat_labels.append(label)
        for i, sim_label in enumerate(sim_labels):
            assert sim_label == feat_labels[i]
        y.extend(sim_data)
        X.extend(feat_data)
    y = np.array(y)
    X = np.array(X)

    print "Mode:", mode
    print "Found {} examples".format(X.shape[0])
    print "Sims Mean: {} (+/-{})".format(y.mean(), 2 * y.std())
    print "Sims min/max: {}/{}".format(np.amin(y), np.amax(y))

    Xcntr = (X - X.mean(axis=0)) / X.std(axis=0)    
    #Xcntr = #Xcntr[0:100,:]
    y = y[:, np.newaxis]

    kern = GPy.kern.rbf(input_dim=Xcntr.shape[1], ARD=True) \
           + GPy.kern.white(Xcntr.shape[1])
    m = GPy.models.GPRegression(Xcntr, y, kern)

    m.unconstrain('')
    m.constrain_positive('')
    
    m.optimize_restarts(num_restarts=20, robust=False, verbose=True,
                        parallel=True, num_processes=8, max_iters=1000)
    
    ypred, yvar, ylow, yhigh = m.predict(Xcntr)


    # Write output
    joblib.dump(m, os.path.join(model_dir, 'gp_reg_model.pkl'))

    params_file = os.path.join(model_dir, 'gp_reg_param.txt')
    with open(params_file, 'w') as pf:
        mvstr = ' '.join([str(v) for v in X.mean(axis=0)])
        pf.write('mean {}\n'.format(mvstr))
        stdvstr = ' '.join([str(v) for v in X.std(axis=0)])
        pf.write('std {}\n'.format(stdvstr))

    with open(pred_file, 'w') as pf:
        pf.write('hour\tstream-id\tsent-d\tpredicted-sim\n') 
        for i in range(y.shape[0]):
            pf.write('{}\t{}\t{}\t{}\n'.format(sim_labels[i][0:13], 
                                               sim_labels[i][12:57],
                                               sim_labels[i][58:],
                                               ypred[i,0]))
            pf.flush()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--sim-file',
                        help=u'Sentence/nugget sim tsv',
                        type=unicode, required=True)

    parser.add_argument('-f', '--feature-file',
                        help=u'Sentence feature tsv',
                        type=unicode, required=True)

    parser.add_argument('-m', '--mode',
                        help=u'mode: max, min, or avg',
                        type=unicode, required=True)

    parser.add_argument('--model-dir',
                        help=u'Location to write gp model params and data',
                        type=unicode, required=True)

    parser.add_argument('--prediction-file',
                        help=u'Write predicted sims here',
                        type=unicode, required=True)

#    parser.add_argument('-o', '--output-file',
#                        help=u'Location to write sims',
#                        type=unicode, required=True)

    args = parser.parse_args()
    sim_file = args.sim_file
    feat_file = args.feature_file
    mode = args.mode
    mdir = args.model_dir
    pfile = args.prediction_file

    if mode not in ['max', 'min', 'avg']:
        sys.stderr.write("Invalid mode: max, min, or avg.\n")
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(mdir):
        os.makedirs(mdir)

    pdir = os.path.dirname(pfile)
    if pdir != u'' and not os.path.exists(pdir):
        os.makedirs(pdir)

    if not os.path.exists(sim_file) or os.path.isdir(sim_file):
        sys.stderr.write((u'--sim-file argument {} either does not exist' \
                          + u' or is a directory!\n').format(sim_file))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(feat_file) or os.path.isdir(feat_file):
        sys.stderr.write((u'--feature-file argument {} either does not exist' \
                          + u' or is a directory!\n').format(feat_file))
        sys.stderr.flush()
        sys.exit()

    return [(sim_file, feat_file)], mode, mdir, pfile

if __name__ == u'__main__':
    main()

