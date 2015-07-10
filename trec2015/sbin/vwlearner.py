import cuttsum.events
import cuttsum.corpora
from cuttsum.l2s import *
from cuttsum.pipeline import InputStreamResource
import pyvw
import pandas as pd
import numpy as np
from datetime import datetime
import random
import matplotlib.pylab as plt
plt.style.use('ggplot')

def evaluate_sequence(actions, dataframes):
    

    all_nuggets = set(
        [n for df in dataframes 
         for ns in df["nuggets"].tolist()
         for n in ns])
    data = []
    results_df = None
    ncache = set()    
    
    d = 0
    sents = [i for i in xrange(len(dataframes[0]))]
    correct_decisions = 0
    n_sents = 0
    for i, a in enumerate(actions, 1):
        if a < len(sents):
            df = dataframes[d].iloc[sents].reset_index(drop=True)
            update = df.iloc[a].to_dict()
            
            gain = df["nuggets"].apply(
                    lambda x: len(x.difference(ncache))).values
            update["max gain"] = max_gain = np.max(gain)
            update["gain"] = gain[a]

            nugget_counts = df["nuggets"].apply(len).values
            update["max nuggets"] = np.max(nugget_counts)
            update["num nuggets"] = nugget_counts[a]
            update["action"] = "select"
            if gain[a] > 0 and max_gain > 0:
                correct_decisions += 1

            update["acc"] = correct_decisions / float(i)   
            ncache.update(update["nuggets"])
            update["rec"] = len(ncache) / float(len(all_nuggets))
            
            n_sents += 1
            update["avg. gain"] = len(ncache) / float(n_sents)

            data.append(update)                            
            del sents[a]

        else:
            if len(sents) > 0:
                df = dataframes[d].iloc[sents].reset_index(drop=True)
                gain = df["nuggets"].apply(
                    lambda x: len(x.difference(ncache))).values
                nc = df["nuggets"].apply(len).values
                max_gain = np.max(gain)
                max_nuggets = np.max(nc)
                if max_gain == 0:
                    correct_decisions += 1
            else:
                max_gain = 0
                max_nuggets = 0
                correct_decisions += 1
            
            update = {
                "timestamp": dataframes[d].iloc[0]["timestamp"],
                "gain": 0, 
                "max gain": max_gain,
                "num nuggets": 0, 
                "max nuggets": max_nuggets,
                "nuggets": set(),
                "action": "next",
                "acc": correct_decisions / float(i),
                "rec": len(ncache) / float(len(all_nuggets)),
                "avg. gain": 0 if n_sents == 0 else len(ncache) / float(n_sents),
            }
            data.append(update)

            d += 1
            if d < len(dataframes):
                sents = [i for i in xrange(len(dataframes[d]))]

    results_df = pd.DataFrame(data,
        columns=["update id", "timestamp", "acc", "rec", "avg. gain", "sent text", "action", "gain", 
                 "max gain", "num nuggets", "max nuggets", "nuggets"])
    return results_df


class PerfectOracle(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.AUTO_HAMMING_LOSS | sch.IS_LDF ) # | sch.AUTO_CONDITION_FEATURES )

    def make_example(self, sent, df, ncache):
        nuggets = [nid for nid in df.iloc[sent]["nuggets"] if nid not in ncache]
        ex = self.example(lambda:
            {"a": [nid for nid in nuggets] if len(nuggets) > 0 else ["none"]},
            labelType=self.vw.lCostSensitive)
        return ex

    def _run(self, (event, docs)):

        ncache = set()
        output = []
        n = 0

        for doc in docs:

            sents = range(len(doc))

            while 1:

                n += 1
                # Create some examples, one for each sentence.
                examples = [self.make_example(sent, doc, ncache) for sent in sents]
                

                # Create a final example for the option "next document".
                # This example has the feature "all_clear" if the max gain
                # of adding any current sentence to the summary is 0.
                # Otherwise the feature "stay" is active.
                gain = doc.iloc[sents]["nuggets"].apply(
                    lambda x: len(x.difference(ncache))).values

                examples.append(
                    self.example(lambda:
                        {"c": ["all_clear" if np.sum(gain) == 0 else "stay"],},
                        labelType=self.vw.lCostSensitive))
               
                # Compute oracle. If max gain > 0, oracle always picks the 
                # sentence with the max gain. Else, oracle picks the last 
                # example which is the "next document" option.
                if len(sents) > 0:
                    oracle = np.argmax(gain) 
                    oracle_gain = gain[oracle] 
                    if oracle_gain == 0:
                        oracle = len(sents)
                else:
                    oracle = 0
                    oracle_gain = 0

                # Make prediction.
                pred = self.sch.predict(
                    examples=examples,
                    my_tag=n,
                    oracle=oracle,
                    condition=[], # (n-1, "p"), ])
                )
                output.append(pred)
                
                if pred < len(sents):
                    ncache.update(doc.iloc[pred]["nuggets"])
                    del sents[pred]
                                        
                elif pred == len(sents):
                    break
   
        return output

class LessPerfectOracle(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.AUTO_HAMMING_LOSS | sch.IS_LDF ) # | sch.AUTO_CONDITION_FEATURES )

    def make_example(self, sent, df, ncache):
        nuggets = df.iloc[sent]["nuggets"]
        ex = self.example(lambda:
            {"a": [nid for nid in nuggets] if len(nuggets) > 0 else ["none"],
             "b": [nid for nid in nuggets if nid in ncache],
            },
            labelType=self.vw.lCostSensitive)
        return ex

    def _run(self, (event, docs)):

        ncache = set()
        output = []
        n = 0

        for doc in docs:

            sents = range(len(doc))

            while 1:

                n += 1
                # Create some examples, one for each sentence.
                examples = [self.make_example(sent, doc, ncache) for sent in sents]
                

                # Create a final example for the option "next document".
                # This example has the feature "all_clear" if the max gain
                # of adding any current sentence to the summary is 0.
                # Otherwise the feature "stay" is active.
                gain = doc.iloc[sents]["nuggets"].apply(
                    lambda x: len(x.difference(ncache))).values

                examples.append(
                    self.example(lambda:
                        {"c": ["all_clear" if np.sum(gain) == 0 else "stay"],},
                        labelType=self.vw.lCostSensitive))
               
                # Compute oracle. If max gain > 0, oracle always picks the 
                # sentence with the max gain. Else, oracle picks the last 
                # example which is the "next document" option.
                if len(sents) > 0:
                    oracle = np.argmax(gain) 
                    oracle_gain = gain[oracle] 
                    if oracle_gain == 0:
                        oracle = len(sents)
                else:
                    oracle = 0
                    oracle_gain = 0

                # Make prediction.
                pred = self.sch.predict(
                    examples=examples,
                    my_tag=n,
                    oracle=oracle,
                    condition=[], # (n-1, "p"), ])
                )
                output.append(pred)
                
                if pred < len(sents):
                    ncache.update(doc.iloc[pred]["nuggets"])
                    del sents[pred]
                                        
                elif pred == len(sents):
                    break
   
        return output

def downsample(training_instances, size=90):
    ds_insts = []
    for event, dataframes in training_instances:
        arange = [x for x in xrange(len(dataframes))]
        random.shuffle(arange)
        arange = arange[:size]
        arange.sort()
        print "downsampling", event.fs_name()
        print arange
        ds_df = [dataframes[i] for i in arange]
        ds_insts.append((event, ds_df))
    return ds_insts


def main(learner, training_ids, test_ids, sample_size, 
         n_iters, report_dir_base):

    extractor = "goose" 
    topk = 20
    delay = None
    threshold = .8
    res = InputStreamResource()

    events = [e for e in cuttsum.events.get_events()
              if e.query_num in training_ids or e.query_num in test_ids]
    training_insts = []
    test_insts = []
    for event in events:
        print "Loading event", event.fs_name()
        corpus = cuttsum.corpora.get_raw_corpus(event)

        # A list of dataframes. Each dataframe is a document with =< 20 sentences.
        # This is the events document stream.
        dataframes = res.get_dataframes(event, corpus, extractor, threshold,
                delay, topk)

        if event.query_num in training_ids:
            training_insts.append((event, dataframes))    
           
        if event.query_num in test_ids:
            test_insts.append((event, dataframes))    

    # Init l2s task.
    vw = pyvw.vw("--search 0 --csoaa_ldf m --search_task hook --ring_size 1024  --quiet  --search_no_caching")

    #task = vw.init_search_task(UpdateSummarizer)
    if learner == "PerfectOracle":
        task = vw.init_search_task(PerfectOracle)
    elif learner == "LessPerfectOracle":
        task = vw.init_search_task(LessPerfectOracle)
    elif learner == "SelectLexNextOracle":
        task = vw.init_search_task(SelectLexNextOracle)
    elif learner == "SelectLexNextLex":
        task = vw.init_search_task(SelectLexNextLex)
    elif learner == "SelectLexNextLexCache":
        task = vw.init_search_task(SelectLexNextLexCache)
    elif learner == "SelectLexGenericNextOracle":
        task = vw.init_search_task(SelectLexGenericNextOracle)
    elif learner == "SelectBasicNextBias":
        task = vw.init_search_task(SelectBasicNextBias)
    elif learner == "SelectBasicNextBiasDocAvg":
        task = vw.init_search_task(SelectBasicNextBiasDocAvg)
    
    for n_iter in range(n_iters):
        print "iter", n_iter + 1
        ds = downsample(training_insts, size=sample_size)
        task.learn(ds)
        all_train_df = [df for inst in training_insts for df in inst[1]]
        feature_weights = task.get_feature_weights(all_train_df)

        write_model(feature_weights, report_dir_base, n_iter)

        for event, dataframes in training_insts:
            # Predict a sequence for this training examples and see if it is sensible.
            print "PREDICTING", event.fs_name()
            sequence, scores = task.predict_with_scores((event, dataframes))
            print sequence
            make_report(event, dataframes, sequence, scores, "train", n_iter,
                report_dir_base)


        for event, dataframes in test_insts:
            # Predict a sequence for this training examples and see if it is sensible.
            print "PREDICTING", event.fs_name()
            sequence, scores = task.predict_with_scores((event, dataframes))
            print sequence
            make_report(event, dataframes, sequence, scores, "test", n_iter,
                report_dir_base)

def write_model(feature_weights, report_dir_base, n_iter):
    dirname = os.path.join(report_dir_base, "iter-{}".format(n_iter + 1))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    path = os.path.join(dirname, "model.tsv")
    with open(path, "w") as f:
        f.write("ns\tfeat\tweight\n")
        for fw in feature_weights:
            f.write("{}\t{}\t{}\n".format(fw[0][:1], fw[0][2:], fw[1]))
    

def make_report(event, dataframes, sequence, scores, part, n_iter, report_dir_base):

    # Run through the sequence of decisions.
    df = evaluate_sequence(sequence, dataframes)
    df = pd.concat([df, scores], axis=1)
    ns = ['a', 'b', 'c', 'd', 'e', 'f']
    l_ns = map(lambda x: "l_" + x, ns)
    o_ns = map(lambda x: "o_" + x, ns)

    cols = [u'acc', u'rec', u'avg. gain', u'action', u'gain', 
            u'max gain', #u'num nuggets', u'max nuggets',
            u'min select score', u'next score',] + l_ns + o_ns
    print df[cols]
                
    report_dir = os.path.join(
        report_dir_base, "iter-{}".format(n_iter + 1), part)
    if not os.path.exists(report_dir): os.makedirs(report_dir)

    results_path = os.path.join(report_dir, event.fs_name() + ".tsv")
    with open(results_path, "w") as f:
        df.to_csv(f, index=False, sep="\t")
    df["timestamp"] = df["timestamp"].apply(datetime.utcfromtimestamp)
    df.set_index("timestamp")[["acc", "rec", "avg. gain"]].plot()
    plt.gcf().suptitle(event.title+ " " + learner + " iter-{}".format(n_iter + 1))
    plt.gcf().savefig(os.path.join(report_dir, "{}.png".format(event.fs_name())))

if __name__ == "__main__":

    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument(u"--learner", type=unicode, choices=[
        u"SelectLexNextOracle", u"SelectLexNextLex", "SelectLexNextLexCache",
        u"SelectLexGenericNextOracle", u"SelectBasicNextBias",
        u"SelectBasicNextBiasDocAvg"],
        help=u"Learner to run.")

    parser.add_argument(u"--training-event-ids", type=int, nargs=u"+",
                        help=u"event ids to select.")
    parser.add_argument(u"--test-event-ids", type=int, nargs=u"+",
                        help=u"event ids to select.")
    parser.add_argument(u"--report-dir",
                        help=u"Where to right outputs.")
    parser.add_argument(u"--num-iters", type=int,
                        help=u"training iters")
    parser.add_argument(u"--sample-size", type=int, default=90,
                        help=u"down sample size")

    args = parser.parse_args()

    n_iters = args.num_iters
    report_dir = args.report_dir
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    training_ids = args.training_event_ids    
    test_ids = args.test_event_ids
    learner = args.learner
    sample_size = args.sample_size

    main(learner, training_ids, test_ids, sample_size, n_iters, report_dir)
