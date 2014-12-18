import argparse
import os
import sys
import multiprocessing as mp
from bs4 import BeautifulSoup
import re
import textwrap

def main():
    html_dir, text_dir = parse_args()
    
    html_files = [fname for fname in os.listdir(html_dir)]

    nfiles = len(html_files)
    cpus = mp.cpu_count()
    files_per_cpu = nfiles / cpus

    jobs = []
    for i in xrange(0, nfiles, files_per_cpu):
        jobs.append((html_dir, text_dir, html_files[i:i+files_per_cpu]))

    pool = mp.Pool(cpus)
    x = pool.map_async(worker, jobs)
    x.get()
    pool.close()
    pool.join()

    print 'Completed processing files in ', html_dir

def worker(args):
    html_dir, text_dir, files = args
    
    for fname in files:
        page = os.path.join(html_dir, fname)
        txtfile = os.path.join(text_dir, fname.replace(u'html', u'txt'))

        soup = BeautifulSoup(open(page))

        div = soup.find("div", {"id": "mw-content-text"})
        with open(txtfile, 'w', 1) as o:
            for tag in div.find_all(True, recursive=False):
                if tag.name == 'p':
                    text = tag.get_text()
                    text = re.sub(r'\[\d+\]', '', tag.get_text())

                    o.write(textwrap.fill(text).encode('utf-8'))
                    o.write('\n\n')



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--html-dir',
                        help=u'HTML directory.',
                        type=unicode, required=True)

    parser.add_argument('-t', '--text-dir',
                        help=u'Output text directory',
                        type=unicode, required=True)

    args = parser.parse_args()
    html_dir = args.html_dir
    text_dir = args.text_dir

    if not os.path.exists(html_dir) or not os.path.isdir(html_dir):
        sys.stderr.write((u'--html-dir argument {} either does not exist' \
                          + u' or is not a directory!\n').format(html_dir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(text_dir):
        os.makedirs(text_dir)


    return html_dir, text_dir

if __name__ == '__main__':
    main()

