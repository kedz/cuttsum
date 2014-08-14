import codecs
import numpy as np
from sklearn.feature_extraction import DictVectorizer

def gold_reader(bow_file, l_file, sim_idx, vector=u'latent'):
    op = codecs.open
    sims = []
    vectors = []
    labels = []
    unicodes = []
    last_hour = None
    with op(bow_file, u'r', u'utf-8') as bf, op(l_file, u'r', u'utf-8') as lf:
        header = lf.readline().strip()  
        
        b_line = bf.readline()
        l_line = lf.readline()

        while b_line and l_line:
            
            b_datum = b_line.strip().split(u'\t')
            b_hour, b_stream_id, b_sent_id, b_unicode = b_datum[0:4]
            bow = {x:1 for x in b_datum[4].split(u' ')}
            
            l_datum = l_line.strip().split(u'\t')
            l_hour, l_stream_id, l_sent_id  = l_datum[0:3]

            sim = float(l_datum[sim_idx])
            lvec = [float(x) for x in l_datum[6:]]
            
            b_label = (b_hour, b_stream_id, b_sent_id)
            l_label = (l_hour, l_stream_id, l_sent_id)
            assert b_label == l_label
            
            if b_hour != last_hour:
                if last_hour is not None:
                    n_points = len(sims)
                    sims = np.array(sims)
                    if vector == u'latent':
                        vectors = np.array(vectors)
                    elif vector == u'bow':
                        vctr = DictVectorizer()
                        vectors = vctr.fit_transform(vectors)
                    unicodes = np.array(unicodes, dtype=(unicode, 1000))
                    yield (last_hour, labels, unicodes, sims, vectors)

                sims = []
                vectors = []
                labels = []
                unicodes = []
                last_hour = b_hour
                
            sims.append(sim)
            if vector == u'latent':
                vectors.append(lvec)
            elif vector == u'bow':
                vectors.append(bow)
            labels.append(b_label)
            unicodes.append(b_unicode)

            b_line = bf.readline()
            l_line = lf.readline()

    if len(vectors) > 0:
        n_points = len(sims)
        sims = np.array(sims)
        if vector == u'latent':
            vectors = np.array(vectors)
        elif vector == u'bow':
            vctr = DictVectorizer()
            vectors = vctr.fit_transform(vectors)
        unicodes = np.array(unicodes, dtype=(unicode, 1000))

        yield (last_hour, labels, unicodes, sims, vectors)

