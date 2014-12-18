import codecs
import os
import argparse
import sys

def main():
    open = codecs.open
    ifile, ofile = parse_cmdline()
    counts = {}
    with codecs.open(ifile, 'r', 'utf-8') as f:
        for line in f:
            tokens = line.strip().split(u' ')
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            
    with open(ofile, 'w', 'utf-8') as f, open(ifile, 'r', 'utf-8') as inpt:
        for line in inpt:
            tokens = line.strip().split(u' ')   
            if len(tokens) == 0:
                continue
            for i, token in enumerate(tokens, 1):
                if counts[token] > 1:
                    f.write(token)
                else:
                    f.write(u'<unk>')
                if i == len(tokens):
                    f.write(u'\n')
                else:
                    f.write(u' ')


def parse_cmdline():

    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input-file',
                        help=u'LM input file.',
                        type=unicode, required=True)

    parser.add_argument('-of', '--output-file',
                        help=u'Output lm input file.',
                        type=unicode, required=True)
 

    args = parser.parse_args()
    ifile = args.input_file
    ofile = args.output_file

    if not os.path.exists(ifile) or os.path.isdir(ifile):
        import sys
        sys.stderr.write((u'Input dir argument {} either does not exist '
                          + u'or is a directory!\n').format(ifile))
        sys.stderr.flush()
        sys.exit()

    odir = os.path.dirname(ofile)
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)

    return ifile, ofile

if __name__ == '__main__':
    main()
