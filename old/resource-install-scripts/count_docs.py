import os
import re

tdir = os.getenv("TREC")
cpath = os.path.join(tdir, "data", "chunks")
opath = os.path.join(tdir, "data", "doc-frequencies")

if not os.path.exists(opath):
    os.makedirs(opath)

for hour in os.listdir(cpath):
    count = 0
    for fname in os.listdir(os.path.join(cpath, hour)):
        if fname == 'index':
            continue
        items = fname.split('-')
        count += int(items[1])
    ofile = os.path.join(opath, '{}.txt'.format(hour))
    with open(ofile, 'w') as f:
        f.write(str(count))
