import codecs
import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import inv
import nltk
from nltk.tokenize import RegexpTokenizer

class WtmfModel:

    def __init__(self, P, v2i, latent_dims=100, lam=20, wm=0.01):
        assert P.shape[0] == latent_dims
        self.P = P
        self.v2i = v2i
        self.latent_dims = latent_dims
        self.lam = lam
        self.wm = wm

    def factor_sentences(self, sents):
        
        data = self.sents2smat(sents)
        return self.factor(data)

    def factor_unicode(self, texts):
        data = self.texts2smat(texts)
        return self.factor(data)

    def factor(self, data):
        q = np.empty((data.shape[0], self.latent_dims), dtype=np.double)

        weights = coo_matrix((np.ones(data.data.shape, dtype=np.double),
                             (data.row, data.col)), shape=data.shape)
        cols = data.shape[0]
        data = data.tocsr()
        weights = weights.tocsr()

        ppte = np.dot(self.P, self.P.T) * self.wm

        for i in range(cols):
            if data.indptr[i+1] - data.indptr[i] == 0:
                q[i,:] = 0
                continue
            pv = self.P[:,data.indices[data.indptr[i]:data.indptr[i+1]]]
            wvals = weights.data[weights.indptr[i]:weights.indptr[i+1]]
            wvals = wvals.reshape((wvals.shape[0], 1))
            A = np.multiply(pv.T,
                            np.tile(wvals, (1, self.latent_dims)) - self.wm)

            v = np.dot(inv(ppte + np.dot(pv,A) + \
                           self.lam * np.eye(self.latent_dims)),
                       np.dot(pv, data.data[data.indptr[i]:data.indptr[i+1]]))
            q[i,:] =  v[:]

        return q                             

    def sents2smat(self, sentences):
        rows = []
        cols = []
        data = []
        
        for row, sent in enumerate(sentences):
            tokens = set([t.token.lower().decode('utf-8') 
                          for t in sent.tokens])
            for t in tokens:
                if t in self.v2i:
                    rows.append(row)
                    cols.append(self.v2i[t])
                    data.append(1)
        nrows = len(sentences)
        vsize = len(self.v2i)
        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)
        return coo_matrix((data, (rows, cols)), shape=(nrows, vsize))         

    def texts2smat(self, texts):
        rows = []
        cols = []
        data = []
        tknzer = RegexpTokenizer(r'\w+')        
        for row, text in enumerate(texts):
            tokens = [t.lower() for t in tknzer.tokenize(text)]
            for t in tokens:
                if t in self.v2i:
                    rows.append(row)
                    cols.append(self.v2i[t])
                    data.append(1)
        nrows = len(texts)
        vsize = len(self.v2i)
        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)
        return coo_matrix((data, (rows, cols)), shape=(nrows, vsize))         
   


def load_model(projection_matrix_txt, vocab_file, latent_dims=100):
    
    v2i = {}
    with codecs.open(vocab_file, 'r', 'utf-8') as vf:
        for idx, line in enumerate(vf):
            v2i[line.strip()] = idx

    with open(projection_matrix_txt, 'r') as pf:
        header_items = pf.readline().strip().split(' ')
        assert len(header_items) == 2
        rows, cols = np.long(header_items[0]), np.long(header_items[1])
        mat = np.empty((rows, cols), dtype=np.double)
        for row, line in enumerate(pf):
            assert row < rows
            elems = [np.double(el) for el in line.strip().split(' ')]
            assert len(elems) == cols
            mat[row,:] = elems

    P = mat.T 


    return WtmfModel(P, v2i, latent_dims=latent_dims)
   
