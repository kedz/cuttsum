#ifndef SRILMCLIENT_H
#define SRILMCLIENT_H

#include <LM.h>
#include <Vocab.h>

class SRILMClient
{
    /* srilm::LMClient takes a char string "port@host" to find ngram server */
    const char *port;

    /* Enable verbose output. */
    int dbg;

    /* Connction to the ngram server */
    LM *lm;

    /* LM Vocabulary */
    Vocab *vocab;

    /* Ngram model order. Default is 3 */
    int order;

    /* Maximum sentence length */
    int maxSentenceLength;

    public:
        SRILMClient(const char *port, bool dbg);
        SRILMClient();
        float sentenceLogProb(char *line);
    private:
        void connect();

};

#endif // SRILMCLIENT_H
