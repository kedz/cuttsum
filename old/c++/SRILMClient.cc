#include "SRILMClient.h"
#include <iostream>
#include <LM.h>
#include <Vocab.h>
#include <LMClient.h>
#include <string.h>

using namespace std;

SRILMClient::SRILMClient(const char *port, bool dbg)
{
    this->dbg = dbg;
    this->port = port;
    this->order = 3;
    this->maxSentenceLength = 1000;
    connect();
}

SRILMClient::SRILMClient()
{
    this->dbg = false;
    this->port = "9986";
    this->order = 3;
    this->maxSentenceLength = 1000;
    connect();
}

void 
SRILMClient::connect()
{
    vocab = new Vocab();
    assert(vocab != 0);

    vocab->unkIsWord() = true;
    vocab->toLower() = false;

    if (dbg)
        cout << "Connecting to ngram server at: " << port << endl;
    lm = new LMClient(*vocab, port, order, order);
   
}

float
SRILMClient::sentenceLogProb(char *line)
{
    char *line_cpy = new char[strlen(line) + 1];
    strcpy(line_cpy, line);
    VocabString sentence[maxSentenceLength + 1];
    unsigned int numWords =
        vocab->parseWords(line_cpy, sentence, maxSentenceLength + 1);
    TextStats sentenceStats;
    LogP lp = lm->sentenceProb(sentence, sentenceStats);
    delete line_cpy;
    return lp;
}
