# language = c++
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, strlen
import socket
import subprocess
import time


cdef extern from "Boolean.h":
    ctypedef bint Boolean

cdef extern from "Vocab.h":
    ctypedef const char *VocabString
    cdef cppclass Vocab:
        Vocab() except +
        Boolean &unkIsWord()
        Boolean &toLower()
        unsigned int parseWords(char *, VocabString *, unsigned int) 

cdef extern from "TextStats.h":
    cdef cppclass TextStats:
        TextStats() except +

cdef extern from "LMClient.h": 
    cdef cppclass LMClient:
        LMClient(Vocab &, const char *, unsigned int, unsigned int) except +
        double sentenceProb(const VocabString *, TextStats &)

def check_status(port):

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(('127.0.0.1', port))
    s.close()
    return result == 0

def start_lm(arpa_path, order, port, timeout=9999):


    cmd = [u'nohup', 'ngram', '-lm', arpa_path, '-tolower', '-order', str(order), '-server-port', 
           str(port)]
        #arpa_path, order, port)] # + u' >lm@port{}.log 2>&1 &'.format(port)]
    log = "lm@port{}".format(port)
    f = open(log, "w")
    proc = subprocess.Popen(cmd, shell=False, stdout=f, stderr=f)

    server_on = False
    start = time.time()
    duration = time.time() - start

    while duration < timeout and server_on is False:
        try:
            server_on = check_status(port)
        except socket.error, e:
            pass

        time.sleep(1)
        duration = time.time() - start
    return proc.pid

cdef class Client:
    cdef LMClient *lmclient
    cdef Vocab *vocab
    cdef object port
    cdef bint to_lower
    cdef unsigned int order
    cdef VocabString[200] sentence  
    cdef bint is_connected

    def __cinit__(self, object port, unsigned int order, bint to_lower):
        self.order = order
        self.port = port
        self.to_lower = to_lower
        self.is_connected = False

    def connect(self):

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if s.connect_ex(("localhost", int(self.port))) != 0:
            s.close()
            raise socket.error(
                "No ngram server running on port {}".format(int(self.port)))
        else:
            s.close()

        self.vocab = new Vocab()
        (&self.vocab.toLower())[0] = self.to_lower
        self.lmclient = new LMClient(
            self.vocab[0], str(self.port), self.order, self.order)
        self.is_connected = True 
        return self

    def __dealloc__(self):
        del self.lmclient
        del self.vocab

    def sentence_log_prob(self, char *line):
        cdef char *line_cpy = <char *> malloc(sizeof(char) * (strlen(line) + 1))
        strcpy(line_cpy, line)
        
        if self.is_connected == False:
            self.connect()

        cdef unsigned int n_words = self.vocab.parseWords(
            line_cpy, self.sentence, 100)
        cdef TextStats *sentstats = new TextStats()
        cdef double lp = self.lmclient.sentenceProb(self.sentence, sentstats[0])
        del sentstats
        free(line_cpy)
        return lp, lp / n_words
