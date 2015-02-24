import numpy as np
from cuttsum.summarizer.filters import APFilteredSummary, \
    APSalienceFilteredSummary, HACFilteredSummary, \
    APSalTRankSalThreshFilteredSummary, RankedSalienceFilteredSummary
import os
from cuttsum.pipeline import jobs
import cuttsum.judgements as cj
import pandas as pd
import random
import subprocess
import re
import multiprocessing


def random_summary(updates, max_length):
    updates_cpy = list(updates)
    random.shuffle(updates_cpy)
    length = 0
    summary = []
    while len(updates_cpy) > 0 and length + len(summary) - 1 < max_length:
        summary.append(updates_cpy.pop())
    summary_text = u'\n'.join(summary)[:max_length]
    return summary_text

def make_rank_summaries(
    rank_sal_cutoff, rank_sim_cutoff, model_summaries, max_samples=1000):
    
    data_dir = os.path.join(os.getenv("TREC_DATA", "."), "rouge")
    config_path = os.path.join(
        data_dir, "rank_sal_{}_sim_{}_config".format(
            rank_sal_cutoff, rank_sim_cutoff))
    config_paths = []
    for job in jobs.event_cross_validation_jobs("crossval"):
        for event, corpus in job.eval_events():
            model_summary = model_summaries[event.fs_name()]
            model_path = os.path.join(
                data_dir, "model_{}".format(event.fs_name()))
            max_len = len(model_summary)
            df = RankedSalienceFilteredSummary().get_dataframe(
                event,  
                rank_sal_cutoff, rank_sim_cutoff)  
            print event.fs_name()
            if df is None:
                print "?"
                import sys
                sys.exit()
            updates = [update.decode('utf-8') 
                       for update in df['text'].tolist()]
            
            for n_sample in xrange(max_samples):
                summary_text = random_summary(updates, max_len)      
                sum_path = os.path.join(
                    data_dir, "rank_sal_{}_sim_{}_sample_{}_{}".format(
                        rank_sal_cutoff, rank_sim_cutoff, n_sample,
                        event.fs_name()))
                with open(sum_path, 'w') as f:
                    f.write(summary_text.encode('utf-8'))
                config_paths.append('{} {}'.format(sum_path, model_path))

    with open(config_path, 'w') as f:
        f.write('\n'.join(config_paths))                

    return config_path



def make_apsaltr_summaries(
    apsal_sal_cutoff, apsal_sim_cutoff, model_summaries, max_samples=1000):
    
    data_dir = os.path.join(os.getenv("TREC_DATA", "."), "rouge")
    config_path = os.path.join(
        data_dir, "apsaltr_sal_{}_sim_{}_config".format(
            apsal_sal_cutoff, apsal_sim_cutoff))
    config_paths = []
    for job in jobs.event_cross_validation_jobs("crossval"):
        for event, corpus in job.eval_events():
            model_summary = model_summaries[event.fs_name()]
            model_path = os.path.join(
                data_dir, "model_{}".format(event.fs_name()))
            max_len = len(model_summary)
            df = APSalTRankSalThreshFilteredSummary().get_dataframe(
                event, job.key, job.feature_set, 
                apsal_sal_cutoff, apsal_sim_cutoff)  
            updates = [update.decode('utf-8') 
                       for update in df['text'].tolist()]
            
            for n_sample in xrange(max_samples):
                summary_text = random_summary(updates, max_len)      
                sum_path = os.path.join(
                    data_dir, "apsaltr_sal_{}_sim_{}_sample_{}_{}".format(
                        apsal_sal_cutoff, apsal_sim_cutoff, n_sample,
                        event.fs_name()))
                with open(sum_path, 'w') as f:
                    f.write(summary_text.encode('utf-8'))
                config_paths.append('{} {}'.format(sum_path, model_path))

    with open(config_path, 'w') as f:
        f.write('\n'.join(config_paths))                

    return config_path


def make_apsal_summaries(apsal_sal_cutoff, apsal_sim_cutoff, 
                         model_summaries, max_samples=1000):
    
    data_dir = os.path.join(os.getenv("TREC_DATA", "."), "rouge")
    config_path = os.path.join(
        data_dir, "apsal_sal_{}_sim_{}_config".format(
            apsal_sal_cutoff, apsal_sim_cutoff))
    config_paths = []
    for job in jobs.event_cross_validation_jobs("crossval"):
        for event, corpus in job.eval_events():
            model_summary = model_summaries[event.fs_name()]
            model_path = os.path.join(
                data_dir, "model_{}".format(event.fs_name()))
            max_len = len(model_summary)
            df = APSalienceFilteredSummary().get_dataframe(
                event, job.key, job.feature_set, 
                apsal_sal_cutoff, apsal_sim_cutoff)  
            updates = [update.decode('utf-8') 
                       for update in df['text'].tolist()]
            
            for n_sample in xrange(max_samples):
                summary_text = random_summary(updates, max_len)      
                sum_path = os.path.join(
                    data_dir, "apsal_sal_{}_sim_{}_sample_{}_{}".format(
                        apsal_sal_cutoff, apsal_sim_cutoff, n_sample,
                        event.fs_name()))
                with open(sum_path, 'w') as f:
                    f.write(summary_text.encode('utf-8'))
                config_paths.append('{} {}'.format(sum_path, model_path))

    with open(config_path, 'w') as f:
        f.write('\n'.join(config_paths))                

    return config_path



def make_ap_summaries(ap_sim_cutoff, model_summaries, max_samples=1000):
    
    data_dir = os.path.join(os.getenv("TREC_DATA", "."), "rouge")
    config_path = os.path.join(
        data_dir, "ap_sim_{}_config".format(ap_sim_cutoff))
    config_paths = []
    for job in jobs.event_cross_validation_jobs("crossval"):
        for event, corpus in job.eval_events():
            model_summary = model_summaries[event.fs_name()]
            model_path = os.path.join(
                data_dir, "model_{}".format(event.fs_name()))
            max_len = len(model_summary)
            df = APFilteredSummary().get_dataframe(
                event, ap_sim_cutoff)  
            updates = [update.decode('utf-8') 
                       for update in df['text'].tolist()]
            
            for n_sample in xrange(max_samples):
                summary_text = random_summary(updates, max_len)      
                sum_path = os.path.join(
                    data_dir, "ap_sim_{}_sample_{}_{}".format(
                        ap_sim_cutoff, n_sample,
                        event.fs_name()))
                with open(sum_path, 'w') as f:
                    f.write(summary_text.encode('utf-8'))
                config_paths.append('{} {}'.format(sum_path, model_path))

    with open(config_path, 'w') as f:
        f.write('\n'.join(config_paths))                

    return config_path


def make_hac_summaries(hac_dist_cutoff, hac_sim_cutoff, 
                       model_summaries, max_samples=1000):
    
    data_dir = os.path.join(os.getenv("TREC_DATA", "."), "rouge")
    config_path = os.path.join(
        data_dir, "hac_dist{}_sim_{}_config".format(
            hac_dist_cutoff, hac_sim_cutoff))
    config_paths = []
    for job in jobs.event_cross_validation_jobs("crossval"):
        for event, corpus in job.eval_events():
            model_summary = model_summaries[event.fs_name()]
            model_path = os.path.join(
                data_dir, "model_{}".format(event.fs_name()))
            max_len = len(model_summary)
            df = HACFilteredSummary().get_dataframe(
                event, hac_dist_cutoff, hac_sim_cutoff)  
            updates = [update.decode('utf-8') 
                       for update in df['text'].tolist()]
            
            for n_sample in xrange(max_samples):
                summary_text = random_summary(updates, max_len)      
                sum_path = os.path.join(
                    data_dir, "hac_dist{}_sim_{}_sample_{}_{}".format(
                        hac_dist_cutoff, hac_sim_cutoff, n_sample,
                        event.fs_name()))
                with open(sum_path, 'w') as f:
                    f.write(summary_text.encode('utf-8'))
                config_paths.append('{} {}'.format(sum_path, model_path))

    with open(config_path, 'w') as f:
        f.write('\n'.join(config_paths))                

    return config_path

def rouge(args):
    config_path = args
    rpath = config_path.replace("config", "result")
    print rpath
    o = subprocess.check_output(
        "cd RELEASE-1.5.5 ; " + \
        "./ROUGE-1.5.5.pl -s -d -n 2 -a -f A -m -z SPL {}".format(config_path),
        shell=True)
    with open(rpath, 'w') as f:
        f.write(o)
    recall = float(re.search('X ROUGE-2 Average_R: ([^ ]+)', o).group(1))
    prec = float(re.search('X ROUGE-2 Average_P: ([^ ]+)', o).group(1))
    f1 = float(re.search('X ROUGE-2 Average_F: ([^ ]+)', o).group(1))
    return config_path, recall, prec, f1


model_summaries = {}
nuggets = pd.concat((cj.get_2013_nuggets(),
                     cj.get_2014_nuggets()))

data_dir = os.path.join(os.getenv("TREC_DATA", "."), "rouge")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
for job in jobs.event_cross_validation_jobs("crossval"):
    for event, corpus in job.eval_events():
        nugget_text = '\n'.join(
            nuggets[nuggets['query id'] == event.query_id]['text'].tolist())
        model_summaries[event.fs_name()] = nugget_text.decode('utf-8')
        model_path = os.path.join(
            data_dir, "model_{}".format(event.fs_name()))
        with open(model_path, 'w') as f:
            f.write(nugget_text)

from cuttsum.misc import ProgressBar
apsal_sal_cutoffs = [.6]
apsal_sim_cutoffs = [.65]

apsaltr_sal_cutoffs = [.6]
apsaltr_sim_cutoffs = [.7]

rank_sal_cutoffs = [1.8]
rank_sim_cutoffs = [.4]

ap_sim_cutoffs = [.7]

hac_dist_cutoffs = [1.7]
hac_sim_cutoffs = [.2]


rank_configs = []
for rank_sal_cutoff in rank_sal_cutoffs:
    print rank_sal_cutoff
    for rank_sim_cutoff in rank_sim_cutoffs:
        print rank_sim_cutoff
        c = make_rank_summaries(
            rank_sal_cutoff, rank_sim_cutoff, model_summaries)
        rank_configs.append(c)

apsaltr_configs = []
for apsaltr_sal_cutoff in apsaltr_sal_cutoffs:
    print apsaltr_sal_cutoff
    for apsaltr_sim_cutoff in apsaltr_sim_cutoffs:
        print apsaltr_sim_cutoff
        c = make_apsaltr_summaries(
            apsaltr_sal_cutoff, apsaltr_sim_cutoff, model_summaries)
        apsaltr_configs.append(c)

apsal_configs = []
for apsal_sal_cutoff in apsal_sal_cutoffs:
    print apsal_sal_cutoff
    for apsal_sim_cutoff in apsal_sim_cutoffs:
        print apsal_sim_cutoff
        c = make_apsal_summaries(
            apsal_sal_cutoff, apsal_sim_cutoff, model_summaries)
        apsal_configs.append(c)

ap_configs = []
for ap_sim_cutoff in ap_sim_cutoffs:
    print ap_sim_cutoff
    c = make_ap_summaries(
        ap_sim_cutoff, model_summaries)
    ap_configs.append(c)

hac_configs = []
for hac_dist_cutoff in hac_dist_cutoffs:
    print hac_dist_cutoff
    for hac_sim_cutoff in hac_sim_cutoffs:
        print hac_sim_cutoff
        c = make_hac_summaries(
            hac_dist_cutoff, hac_sim_cutoff, model_summaries)
        hac_configs.append(c)

def print_results(configs):

    n_jobs = len(configs)
    pb = ProgressBar(n_jobs)
    results = []
    for result in multiprocessing.Pool(5).imap_unordered(rouge, configs):
        pb.update()
        results.append(result)
    results.sort(key=lambda x: x[3], reverse=True)
    for result in results[:10]:
        print result

print "BEST RANK"
print_results(rank_configs + apsaltr_configs + apsal_configs + ap_configs + \
    hac_configs)
