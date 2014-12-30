import cuttsum.sc
from cuttsum.misc import ProgressBar
import os
from multiprocessing import Pool
import streamcorpus as sc
import subprocess


def worker(path):
    basename = os.path.basename(path)
    checksum = basename.split('.')[0]
    checksum = checksum.split('-')[-1]
    actual_checksum = subprocess.check_output(
        "xzcat {} | md5sum".format(path), shell=True).split(" ")[0] 
    if checksum != actual_checksum:
        print "Bad path:", path

paths = []
for path, dirs, files in os.walk(cuttsum.sc.SCChunkResource().dir_):
    for fname in files:
        paths.append(os.path.join(path, fname))

pool = Pool(10)
pb = ProgressBar(len(paths))

for result in pool.imap_unordered(worker, paths):
    pb.update()
