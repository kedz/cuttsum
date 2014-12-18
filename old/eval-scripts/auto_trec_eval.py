import os
import codecs
from datetime import datetime
import cuttsum.wtmf as wtmf
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np
import argparse

def main():
    
    updates_file, model_file, ssim_model, ssim_vocab, ofile = parse_args()
    
    update_timestamps, update_texts = load_updates(updates_file)
    nugget_timestamps, nugget_texts = load_nugget_updates(model_file)
    wtmf_model = wtmf.load_model(ssim_model, ssim_vocab)
    update_lvecs = wtmf_model.factor_unicode(update_texts)
    nugget_lvecs = wtmf_model.factor_unicode(nugget_texts)

    S = cosine_similarity(update_lvecs, nugget_lvecs)
#    S = np.ma.masked_array(S, mask=(S <= 0))
    S[np.where(S < .60)] = 0.0

    S[np.where(S > 0.0)] = 1.0
    
    L = compute_latency_discount_matrix(update_timestamps, nugget_timestamps)

    gain = 0
    gain_discounted = 0

    rows = np.argmax(S, axis=0)

    for r, c in zip(rows, np.arange(S.shape[1])):
        gain += S[r,c]
        gain_discounted += S[r,c] * L[r,c]

    n_updates = len(update_texts)
    n_nuggets = len(nugget_texts)
#    IN = np.argmax(Spos, axis=1)
#    IU = np.arange(0, n_updates)
    #Spo
    #Gf = Spos[IU,IN]
    #Gl = Gf * L[IU,IN]


    exp_gain_f = gain / float(n_updates)
    exp_gain_l = gain_discounted / float(n_updates)
    comprehensiveness_f = gain / float(n_nuggets)
    comprehensiveness_l = gain_discounted / float(n_nuggets)
    if exp_gain_f > 0:
        fmeas_f = 2*exp_gain_f * comprehensiveness_f / \
            (exp_gain_f + comprehensiveness_f)
    else:
        fmeas_f = 0
    if exp_gain_l > 0:
        fmeas_l = 2*exp_gain_l * comprehensiveness_l / \
            (exp_gain_l + comprehensiveness_l)
    else:
        fmeas_l = 0

    with open(ofile, u'w') as f:
        f.write("Discount\tComprehensiveness\tE[gain]\tF-measure\n")
        f.write("{}\t{}\t{}\t{}\n".format(
            "Discount-Free", comprehensiveness_f, exp_gain_f, fmeas_f))
        f.write("{}\t{}\t{}\t{}\n".format(
            "Latency-Discounted", comprehensiveness_l, exp_gain_l, fmeas_l))
        f.flush()
    print "{}\t{}\t{}\t{}".format(
        "Discount-Free", comprehensiveness_f, exp_gain_f, fmeas_f)
    print "{}\t{}\t{}\t{}".format(
        "Latency-Discounted", comprehensiveness_l, exp_gain_l, fmeas_l)

def compute_latency_discount_matrix(update_timestamps, nugget_timestamps):
    n_rows = len(update_timestamps)
    n_cols = len(nugget_timestamps)

    L = np.empty((n_rows, n_cols))

    for r in range(n_rows):
        for c in range(n_cols):
            dt = update_timestamps[r] - nugget_timestamps[c]
            L[r,c] = dt.total_seconds()
    alpha = 3600.0 * 6.0
    L = 1.0 - (2.0 / np.pi) * np.arctan(L / alpha)
    return L

def load_updates(updates_file):

    timestamps = []
    texts = []
    with codecs.open(updates_file, u'r', u'utf-8') as f:
        for line in f:
            #timestamp, stream_id, sent_id, text = line.strip().split(u'\t')
            timestamp, score, text = line.strip().split(u'\t')
            timestamp = datetime.strptime(timestamp, u'%Y-%m-%d-%H-%M-%S')
            timestamps.append(timestamp)
            texts.append(text)
    return timestamps, texts      

def load_nugget_updates(updates_file):

    timestamps = []
    texts = []
    with codecs.open(updates_file, u'r', u'utf-8') as f:
        for line in f:
            timestamp, text = line.strip().split(u'\t')
            timestamp = datetime.strptime(timestamp, u'%Y-%m-%d-%H-%M-%S')
            timestamps.append(timestamp)
            texts.append(text)
    return timestamps, texts     

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--sys-updates-file',
                        help=u'system updates file',
                        type=unicode, required=True)

    parser.add_argument('-m', '--mod-updates-file',
                        help=u'model updates file',
                        type=unicode, required=True)
 
    parser.add_argument('--sent-sim-model',
                        help=u'sentence sim model',
                        type=unicode, required=True)
 
    parser.add_argument('--sent-sim-vocab',
                        help=u'sentence sim vocab',
                        type=unicode, required=True)
 
    parser.add_argument('-o', '--output-file',
                        help=u'output file',
                        type=unicode, required=True)
 
    args = parser.parse_args()
    sys_updates = args.sys_updates_file
    mod_updates = args.mod_updates_file
    ssim_model = args.sent_sim_model
    ssim_vocab = args.sent_sim_vocab
    ofile = args.output_file

    if not os.path.exists(sys_updates) or os.path.isdir(sys_updates):
        sys.stderr.write((u'--sys-updates-file argument {} either does not ' \
                          u'exist or is a directory!\n').format(sys_updates))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(mod_updates) or os.path.isdir(mod_updates):
        sys.stderr.write((u'--mod-updates-file argument {} either does not ' \
                          u'exist or is a directory!\n').format(mod_updates))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(ssim_model) or os.path.isdir(ssim_model):
        sys.stderr.write((u'--sent-sim-model argument {} either does not ' \
                          u'exist or is a directory!\n').format(ssim_model))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(ssim_vocab) or os.path.isdir(ssim_vocab):
        sys.stderr.write((u'--sent-sim-vocab argument {} either does not ' \
                          u'exist or is a directory!\n').format(ssim_vocab))
        sys.stderr.flush()
        sys.exit()

    odir = os.path.dirname(ofile)
    if odir != u'' and not os.path.exists(odir):
        os.makedirs(odir)

    return sys_updates, mod_updates, ssim_model, ssim_vocab, ofile

if __name__ == u'__main__':
    main()
