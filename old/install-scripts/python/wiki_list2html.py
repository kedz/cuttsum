import os
import sys
import argparse
import requests
import codecs
import multiprocessing as mp
import urllib

def main():
    odir, wiki_list, depth, cpus, ldir = parse_args()
    pages = read_list(wiki_list, depth)

    npages = len(pages)
    pages_per_cpu = npages / cpus
    jobs = []
    pid = 0

    log_prefix = os.path.split(wiki_list)[-1].replace('.txt', '_{}.log')


    for i in xrange(0, npages, pages_per_cpu):
        lfile = os.path.join(ldir, log_prefix.format(pid))
        jobs.append((odir, pages[i:i+pages_per_cpu], lfile))
        pid += 1

    pool = mp.Pool(cpus)
    x = pool.map_async(worker, jobs)
    x.get()
    pool.close()
    pool.join()


def worker(args):
    odir, pages, log_file = args
    npages = len(pages)
 
    lf = codecs.open(log_file, 'w', 'utf-8')


    for i, page in enumerate(pages, 1):

        lf.write(u'Requesting: {}\n'.format(page))
        lf.flush()
        req = {'action':'query', 'format':'json', 'prop':'revisions',
               'titles':page, 
               'rvprop':'timestamp|ids',
               'rvstart':'20111001000000', 'rvdir':'older' }

        result = requests.get('http://en.wikipedia.org/w/api.php',
                              params=req).json()

        pid = result['query']['pages'].keys()[0]
        revs = result['query']['pages'][pid].get('revisions', None)
        if revs is not None:
            if len(revs) > 0:
                rev = revs[0]

                url_tmp = u'http://en.wikipedia.org/w/index.php?' \
                          + u'title={}&oldid={}'
                url = url_tmp.format(page.replace(u' ', u'_'),
                                     rev['parentid'])

                ofile = u'{}_{}_{}.html'.format(rev['timestamp'],
                                                rev['parentid'],
                                                page).replace(u' ', u'_')
                ofile = ofile.replace(u'/', u'_') 
                ofile = ofile.replace(u'\\', u'_') 
                #ofile = ofile.encode('utf-8')
                opath = os.path.join(odir, ofile)
                opath = opath.encode('utf-8')
                try:
                    req = requests.get(url)
                except requests.exceptions.InvalidURL, e:
                    print "Invalid url?", url
                    continue
                with codecs.open(opath, 'w', 'utf-8') as f: f.write(req.text)

                lf.write(u'{}/{}) {} {}\n'.format(i, npages, rev['timestamp'], 
                                                  rev['parentid']))
                lf.write(u'\t--> {} {}k\n'.format(ofile,
                                                  os.path.getsize(opath)/1024))
                lf.flush()
        else:
            lf.write(u'{} has no revision before cutoff date.\n'.format(page))
            lf.flush()
    lf.close()

def read_list(wiki_list, depth):
    pages = []
    with codecs.open(wiki_list, 'r', 'utf-8') as f:
        for line in f:
            items = line.strip().split(u'\t')
            if len(items) != 3:
                continue
            item_depth = int(items[1])
            if item_depth <= depth:
                pages.append(items[2])
    return pages

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--html-dir',
                        help=u'Location to write html files.',
                        type=unicode, required=True)

    parser.add_argument('-w', '--wiki-list',
                        help=u'TSV of pageids, depth, page title.',
                        type=unicode, required=True)

    parser.add_argument('-m', '--max-depth',
                        help=u'Max depth to retreive',
                        type=int, required=False, default=5)

    parser.add_argument('-p', '--num-processes',
                        help=u'Number of processes to use',
                        type=int, required=False, default=1)

    parser.add_argument('-l', '--log-dir',
                        help=u'Location to write logs',
                        type=str, required=False, default='logs')



    args = parser.parse_args()
    odir = args.html_dir
    wiki_list = args.wiki_list
    depth = args.max_depth
    cpus = args.num_processes
    ldir = args.log_dir

    if ldir != '' and not os.path.exists(ldir):
        os.makedirs(ldir)

    if not os.path.exists(wiki_list) or os.path.isdir(wiki_list):
        sys.stderr.write((u'--wiki-list argument {} either does not exist' \
                          + u' is a directory!\n').format(wiki_list))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(odir):
        os.makedirs(odir)

    return odir, wiki_list, depth, cpus, ldir

if __name__ == '__main__':
    main()
