import cuttsum.judgements as cj
import cuttsum.events
import cuttsum.submissions as cs
from collections import defaultdict
import numpy as np
import os
import sys
import gzip

def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1] / float(len(source))

def main(ostream):

    all_matches = cj.get_2014_matches()
    all_sampled_updates = cs.get_2014_sampled_updates()
    all_extended_updates = cs.get_2014_sampled_updates_extended()
    all_lev_updates = cs.get_2014_sampled_updates_levenshtein()
    for event in cuttsum.events.get_2014_events():
        matches = all_matches[all_matches[u'query id'] == event.query_id]
        sampled_updates = \
            all_sampled_updates[
                all_sampled_updates['query id'] == event.query_id]
        sampled_updates_ext = \
            all_extended_updates[
                all_extended_updates['query id'] == event.query_id]
        sampled_updates_lev = \
            all_lev_updates[
                all_lev_updates['query id'] == event.query_id]
       
         
        text2nugget = defaultdict(set)
        used_uids = set()
        for name, update in sampled_updates_ext.iterrows():
            update_matches = \
                matches[matches[u'update id'] == update[u'update id']]
            n_matches = len(update_matches)
            
            if n_matches > 0:
                for _, update_match in update_matches.iterrows():
                    start = update_match[u'match start']
                    end = update_match[u'match end']
                    text2nugget[update[u'text']].add((start, end, update_match[u'nugget id']))
                    used_uids.add(update_match[u'update id'])
                    print "{}\t{}\t{}\t{}\t{}\t0".format(
                        event.query_id, update[u'update id'], update_match[u'nugget id'], start, end)
                    ostream.write("{}\t{}\t{}\t{}\t{}\t0\n".format(
                        event.query_id, update[u'update id'], update_match[u'nugget id'], start, end))


        n_ext = len(sampled_updates_ext)
        for index, (_, update) in enumerate(sampled_updates_ext.iterrows(), 1):
            sys.stdout.write("Finding exact & fuzzy matches %{:0.3f}\r".format(
                100. * float(index) / n_ext))
            sys.stdout.flush()
            if update[u'update id'] in used_uids:
                continue
            if update[u'text'] in text2nugget:
                for start, end, nugget_id in text2nugget[update[u'text']]:
                    print "{}\t{}\t{}\t{}\t{}\t0".format(
                        event.query_id, update[u'update id'], nugget_id, start, end)
                    ostream.write("{}\t{}\t{}\t{}\t{}\t0\n".format(
                        event.query_id, update[u'update id'], nugget_id, start, end))
            else:
                unique_matches = set()
                for text in text2nugget.keys():
                    tgt = update[u'text']
                    src = text
                    if len(src) > len(tgt):
                        src = update[u'text']
                        tgt = text
                    diff = len(tgt) - len(src)
                    if diff / float(len(tgt)) > .2:
                        continue
                                                
                    dist = levenshtein(update[u'text'].lower(), text.lower())
                    if dist <= .2:
                        for start, end, nugget_id in text2nugget[text]:
                            
                            if nugget_id not in unique_matches:
                                print "{}\t{}\t{}\t{}\t{}\t{}".format(
                                    event.query_id, update[u'update id'], nugget_id, '?', '?', dist)
                                ostream.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                    event.query_id, update[u'update id'], nugget_id, '?', '?', dist))
                                unique_matches.add(nugget_id)
            used_uids.add(update[u'update id'])

if __name__ == u'__main__':
    outdir = os.getenv(u"TREC_DATA", None)
    if outdir is None:
        outdir = os.path.join(os.getenv(u'HOME', u'.'), u'trec-data')
        print u'TREC_DATA env. var not set. Using {} to store data.'.format(
            outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)           
    path = os.path.join(outdir, u'matches.extended.tsv.gz')
    with gzip.open(path, u'w') as f:
        main(f)  
