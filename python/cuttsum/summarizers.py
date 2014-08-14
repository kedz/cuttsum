import os
import codecs
from cuttsum.util import hour_str2datetime_interval 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AffinityPropagation

class SummarizerBase(object):

    def __init__(self, use_temp=False, vec_dims=100):
        self.n_updates_ = 0
        self.use_temp_ = use_temp
        self.vec_dims_ = vec_dims
        self.update_vecs_ = np.empty((5000, self.vec_dims_))
        self.update_sent_ids_ = np.empty((5000,), dtype=int)
        self.update_stream_ids_ = np.empty((5000,), dtype=(str, 43))
        self.update_texts_ = np.empty((5000,), dtype=('U', 1000))
        self.update_timestamps_ = np.empty((5000,), dtype=(str, 19))
        self.update_saliences_ = np.empty((5000,))
        if self.use_temp_ is True:
            self.update_saliences_pen_ = np.empty((5000,))


    def add_updates(self, update_idxs, time_int, labels, unicodes, saliences, saliences_pen, X):
        for i, e in enumerate(update_idxs):
            up_idx = i + self.n_updates_
            self.update_vecs_[up_idx,:] = X[e,:]
            self.update_texts_[up_idx] = unicodes[e]
            self.update_timestamps_[up_idx] = time_int.stop.strftime(u'%Y-%m-%d-%H-%M-%S')
            self.update_stream_ids_[up_idx] = labels[e][1]
            self.update_sent_ids_[up_idx] = labels[e][2]
            self.update_saliences_[up_idx] = saliences[e]
            if saliences_pen is not None:
                self.update_saliences_pen_[up_idx] = saliences_pen[e]
        self.n_updates_ += len(update_idxs)

    def penalize_salience(self, saliences, X, mode=u'agg', scale=1.0):

        n_points = X.shape[0]
        up_X = self.update_vecs_[0:self.n_updates_,:]
        S = cosine_similarity(X, up_X)
        dup_coords = np.where(S>.95) 
        S[dup_coords[0],:] = 0
        S[np.where(S<.15)] = 0 

        norms = np.sum(S, axis=0)
        # Avoid divide by zero error if we have column of 0's.
        norms[np.where(norms==0)] = 1

        Snorm = S / norms
        update_saliences = self.update_saliences_[0:self.n_updates_]
        update_saliences = update_saliences[:,np.newaxis]
        if mode == u'agg':
            Penalties = np.sum(Snorm * update_saliences.T, axis=1)
        elif mode == u'max':
            I = np.argmax(Snorm, axis=1)
            Penalties = Snorm[np.arange(n_points),I] * update_saliences[I].reshape((n_points,))

        new_saliences = saliences - scale * Penalties
        # Set any sentences that have already been selected as an updates to 
        # the minimum salience. 
        new_saliences[dup_coords[0]] = np.amin(new_saliences)
        return new_saliences.reshape((n_points,))

    def unique_indices(self, unicodes, saliences, return_counts=False):
        
        u, index, assignments = np.unique(
            unicodes, return_index=True, return_inverse=True)
        
        n_unique = u.shape[0]
        index_set = np.empty((n_unique,), dtype=int)
        counts = np.zeros((n_unique,), dtype=int)
            

        for i in range(n_unique):
            II = np.where(assignments == i)[0]
            index_set[i] = II[np.argmax(saliences[II])]
            counts[i] = II.shape[0]
        if return_counts is True:
            return index_set, counts
        else:
            return index_set

    def write_updates(self, odir):
    
        updates_file = os.path.join(odir, u'updates.txt')
        with codecs.open(updates_file, u'w', u'utf-8') as uf:    
            for i in range(self.n_updates_):
                uf.write(u'{}\t{}\t{}\n'.format(
                    self.update_timestamps_[i], self.update_saliences_[i], 
                    self.update_texts_[i]))
            uf.flush()

    def write_iterative_summary(self, odir, hour):
        summ_file = os.path.join(odir, u'sum@{}.txt'.format(hour))
        with codecs.open(summ_file, u'w', u'utf-8') as uf:    
            for i in range(self.n_updates_):
                uf.write(u'{}\n'.format(self.update_texts_[i]))
            uf.flush()


class RankSummarizer(SummarizerBase):
    
    def run(self, data_reader, odir, n_return, penalty_mode):

        log_handle, log_system = logger(odir)
        saliences_pen = None
        for hour, labels, unicodes, saliences, X in data_reader:

            time_int = hour_str2datetime_interval(hour)
            n_points_total = X.shape[0]

            ### DEDUP ###        
            I = self.unique_indices(unicodes, saliences)
            unicodes = unicodes[I]
            saliences = saliences[I]
            X = X[I,:]
            labels = [labels[idx] for idx in I] 
            n_points = X.shape[0]

            print "System time: {} -- {}".format(time_int.start, time_int.stop)
            print "Received {} unique sentences from {} total".format(
                n_points, n_points_total)

            if self.use_temp_ is True and self.n_updates_ > 0:
                saliences_pen = self.penalize_salience(
                    saliences, X, penalty_mode)
                ranks = saliences_pen
            else:
                ranks = saliences

            sorted_idxs = sorted(range(n_points),
                                 key=lambda x: ranks[x], 
                                 reverse=True)
            update_idxs = sorted_idxs[0:n_return]

            for e in update_idxs:
                if saliences_pen is not None:
                    print saliences[e], saliences_pen[e], unicodes[e].encode(u'utf-8')
                else:
                    print saliences[e], unicodes[e].encode(u'utf-8')
            print


            self.add_updates(
                update_idxs, time_int, labels,
                unicodes, saliences, saliences_pen, X)

            self.write_iterative_summary(
                odir, time_int.stop.strftime(u'%Y-%m-%d-%H'))

            log_system(
                sorted_idxs, time_int, labels, 
                unicodes, saliences, saliences_pen)
             
        self.write_updates(odir)
        log_handle.close()

class APSummarizer(SummarizerBase):

    def run(self, data_reader, odir, penalty_mode):
        
        log_handle, log_system = logger(odir)
        saliences_pen = None

        for hour, labels, unicodes, saliences, X, in data_reader:
            time_int = hour_str2datetime_interval(hour)
            n_points_total = X.shape[0]

            ### DEDUP AND COUNT DUPLICATES ###
            I, counts = self.unique_indices(
                unicodes, saliences, return_counts=True)
            unicodes = unicodes[I]
            saliences = saliences[I]
            X = X[I,:]
            labels = [labels[idx] for idx in I] 
            n_points = X.shape[0]

            print "System time: {} -- {}".format(
                time_int.start, time_int.stop)
            print "Received {} unique sentences from {} total".format(
                n_points, n_points_total)

            if self.use_temp_ is True and self.n_updates_ > 0:
                saliences_pen = self.penalize_salience(
                    saliences, X, penalty_mode, scale=100.0)
                ranks = saliences_pen
            else:
                ranks = saliences

            ### Init Preferences and Similarities ###
            P = self.compute_preferences(ranks, n_points, counts)
            A = self.compute_affinities(X, P, counts)

            af = AffinityPropagation(
                preference=P, affinity='precomputed', max_iter=1000,
                damping=.7, verbose=True).fit(A)

            exemplars = af.cluster_centers_indices_
            assignments = exemplars[af.labels_]

            update_idxs = [e for e in exemplars 
                           if saliences[e] > 2.0]
                           #if np.where(assignments == e)[0].shape[0] - 1 > 0 \
                           #and saliences[e] > 2.0]

            sorted_idxs = []
            for e in exemplars:
                sorted_idxs.append(e)
                for m in np.where(assignments == e)[0]:
                    if e != m:
                        sorted_idxs.append(e)

            self.add_updates(
                update_idxs, time_int, labels,
                unicodes, saliences, saliences_pen, X)

            self.write_iterative_summary(
                odir, time_int.stop.strftime(u'%Y-%m-%d-%H'))

            log_system(
                sorted_idxs, time_int, labels, 
                unicodes, saliences, saliences_pen)

            for e in update_idxs:
                if saliences_pen is not None:
                    print saliences[e], saliences_pen[e],
                    print unicodes[e].encode(u'utf-8')
                else:
                    print saliences[e], unicodes[e].encode(u'utf-8')
            print

        self.write_updates(odir)

    def compute_preferences(self, saliences, n_points, counts):
        saliences = saliences.reshape(n_points, 1)
        amax = np.minimum(-7 + np.exp(-np.log(n_points/100.0)), -2)
        scaler = MinMaxScaler(feature_range=(-9, amax))
        return scaler.fit_transform(saliences)

    def compute_affinities(self, X, P, counts):
        S = cosine_similarity(X)
        mask = np.logical_or(np.eye(S.shape[0], dtype=bool), S < .2)
        Sma = np.ma.masked_array(S, mask=mask)
        scaler = MinMaxScaler(feature_range=(-3, -1))
        return scaler.fit_transform(Sma)
   


def logger(odir):
    import gzip
    log_file = os.path.join(odir, u'log.txt.gz')
    lf = gzip.open(log_file, u'wb')
    def log_system(sorted_idxs, time_int, labels, unicodes, saliences, saliences_pen):
        lf.write('System time: {} -- {}\n'.format(time_int.start, time_int.stop))
        for idx in sorted_idxs:
            ouni = u'{}\t{}\t{}\t{}\t{}\n'.format(
                labels[idx][1], labels[idx][2], saliences[idx], saliences_pen[idx] if saliences_pen is not None else 'NaN', unicodes[idx])
            lf.write(ouni.encode(u'utf-8'))
            lf.flush()
    return lf, log_system
  
