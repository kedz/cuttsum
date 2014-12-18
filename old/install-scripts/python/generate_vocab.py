import codecs
import os
import argparse
import sys

def main():
    ifiles, ofile, t = parse_cmdline()
    counts = {}
    for ifile in ifiles:
        with codecs.open(ifile, 'r', 'utf-8') as f:
            for line in f:
                tokens = line.strip().split(u' ')
                for token in tokens:
                    counts[token] = counts.get(token, 0) + 1

    with codecs.open(ofile, 'w', 'utf-8') as o:
        for token in counts.iterkeys():
            if counts[token] >= t:
                o.write(token)
                o.write(u'\n')
                o.flush()
                

def parse_cmdline():

    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input-files',
                        help=u'LM input files.',
                        type=unicode, nargs='+', required=True)

    parser.add_argument('-of', '--output-file',
                        help=u'Output lm input file.',
                        type=unicode, required=True)
 
    parser.add_argument('-t', '--threshold',
                        help=u'Replace words occurring less than -c times.',
                        type=int, required=True)
 

    args = parser.parse_args()
    ifiles = args.input_files
    ofile = args.output_file
    t = args.threshold

    if t < 1:
        t = 1
        sys.stderr.write(u'Warning: Illegal threshold, using 1 instead.\n')
        sys.stderr.flush()

    for ifile in ifiles:
        if not os.path.exists(ifile) or os.path.isdir(ifile):
            import sys
            sys.stderr.write((u'Input dir argument {} either does not exist '
                              + u'or is a directory!\n').format(ifile))
            sys.stderr.flush()
            sys.exit()

    odir = os.path.dirname(ofile)
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)

    return ifiles, ofile, t

if __name__ == '__main__':
    main()
