import os
import sys
import argparse
from cuttsum.readers import gold_reader
from cuttsum.summarizers import APSummarizer

def main():
    args = parse_args()
    bow_file, lvec_file, sim_file = args[u'input_files']
    odir = args[u'output']
    dbg_sim_mode, use_temp, penalty_mode = args[u'params']
    
    if sim_file is None:
        if dbg_sim_mode == u'max':
            sim_idx = 3
        elif dbg_sim_mode == u'min':
            sim_idx = 4
        elif dbg_sim_mode == u'avg':
            sim_idx = 5
        data_reader = gold_reader(bow_file, lvec_file, sim_idx)

    else:
        print "IMPLEMENT SIM FILE LOADER"
        sys.exit()

    ts_system = APSummarizer(use_temp, vec_dims=100)
    ts_system.run(data_reader, odir, penalty_mode)
    
    print "Run complete!"

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('-b', '--bow-file',
                        help=u'BOW file',
                        type=unicode, required=True)

    parser.add_argument('-l', '--lvec-file',
                        help=u'latent vector file',
                        type=unicode, required=True)

    parser.add_argument('--debug-sim-mode',
                        help=u'max, min, or avg',
                        type=unicode, 
                        default=u'max',
                        required=False)

    parser.add_argument('-s', '--sim-file',
                        help=u'sim file',
                        type=unicode, required=False)

    parser.add_argument('-o', '--output-dir',
                        help=u'Summary output location',
                        type=unicode, required=True)

#    parser.add_argument('-p', '--const-pref',
#                        help=u'Use a constant preference value',
#                        type=int, required=False)

    parser.add_argument('--temp', dest='use_temp', action='store_true')
    parser.add_argument('--no-temp', dest='use_temp', action='store_false')
    parser.set_defaults(use_temp=True)  

    parser.add_argument('-p', '--penalty-mode',
                        help=u'agg or max',
                        type=unicode, required=True)

  
    args = parser.parse_args()
    bow_file = args.bow_file
    lvec_file = args.lvec_file
    sim_file = args.sim_file
    odir = args.output_dir
#    ofile = args.output_file
#    eng_log_file = args.energy_log
    dbg_sim_mode = args.debug_sim_mode
#    const_pref = args.const_pref
#    rouge_dir = args.rouge_dir
#    cfile = args.cluster_file
    use_temp = args.use_temp
    penalty_mode = args.penalty_mode

    if use_temp is True and penalty_mode not in [u'agg', u'max']:
        import sys
        sys.stderr.write(u'Bad --penalty-mode argument: \'agg\' or \'max\'\n')
        sys.stderr.flush()
        sys.exit()


#    if rouge_dir != '' and not os.path.exists(rouge_dir):
#        os.makedirs(rouge_dir)

    if dbg_sim_mode not in [u'max', u'min', u'avg']:
        sys.stderr.write(u'Bad argument for --debug-sim-mode: ')
        sys.stderr.write(u'max, min or avg are legal args\n')
        sys.stderr.flush()
        sys.exit()

#    eldir = os.path.dirname(eng_log_file)
#    if eldir != u'' and not os.path.exists(eldir):
#        os.makedirs(eldir)

#    cdir = os.path.dirname(cfile)
#    if cdir != '' and not os.path.exists(cdir):
#        os.makedirs(cdir)


#    odir = os.path.dirname(ofile)
    if odir != u'' and not os.path.exists(odir):
        os.makedirs(odir)

    if not os.path.exists(bow_file) or os.path.isdir(bow_file):
        sys.stderr.write((u'--bow-file argument {} either does not exist' \
                          + u' or is a directory!\n').format(bow_file))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(lvec_file) or os.path.isdir(lvec_file):
        sys.stderr.write((u'--lvec-file argument {} either does not exist' \
                          + u' or is a directory!\n').format(lvec_file))
        sys.stderr.flush()
        sys.exit()

    if sim_file is not None:
        if not os.path.exists(sim_file) or os.path.isdir(sim_file):
            sys.stderr.write((u'--sim-file argument {} either does not exist' \
                              + u' or is a directory!\n').format(sim_file))
            sys.stderr.flush()
            sys.exit()

    args_dict = {u'input_files': [bow_file, lvec_file, sim_file],
                 u'output': odir,
                 u'params': [dbg_sim_mode, use_temp, penalty_mode]}
    return args_dict

if __name__ == '__main__':
    main()
