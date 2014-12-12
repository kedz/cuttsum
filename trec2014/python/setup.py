import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def setup_package_data():
    data = [os.path.join(u'2014-data', u'trec2014-ts-topics-test.xml'),
            os.path.join(u'2013-data', u'trec2013-ts-topics-test.xml'),
            os.path.join(u'2013-data',
                u'kba-streamcorpus-2013-v0_2_0' + \
                u'-english-and-unknown-language.s3-paths.txt.gz'),
            os.path.join(u'2013-data',
                'kba-streamcorpus-2013-v0_2_0.s3-paths.txt.gz'),
            os.path.join(u'2014-data', u'nuggets.tsv.gz'),
            os.path.join(u'2014-data', u'matches.tsv.gz'),
            os.path.join(u'2014-data', u'updates_sampled.tsv.gz'),
            os.path.join(u'2014-data', u'updates_sampled.extended.tsv.gz'),
            os.path.join(u'2014-data', u'updates_sampled.levenshtein.tsv.gz'),
            os.path.join(u'2014-data', 
                u'*.gz'),
            os.path.join(u'2014-data', 
                u'streamcorpus-2014-v0_3_0-ts-filtered.s3-paths.txt.gz'),
            os.path.join(u'2014-data', u'cunlp-updates.ssv.gz'),
            os.path.join(u'models'),
            os.path.join(u'models', u'article_clf.pkl'),      
            os.path.join(u'models', u'article_clf.pkl_01.npy'),
            os.path.join(u'models', u'article_clf.pkl_02.npy'),
            os.path.join(u'models', u'article_clf.pkl_03.npy'),
            os.path.join(u'models', u'article_clf.pkl_04.npy'),
            os.path.join(u'models', u'article_clf.pkl_05.npy'),
            os.path.join(u'models', u'article_vectorizer.pkl')]
         
    return {'cuttsum': data}


setup(
    name = "cuttsum",
    version = "0.1.1",
    author = "Chris Kedzie",
    author_email = "kedzie@cs.columbia.edu",
    description = ("Code repo for Columbia U. TREC Temporal Summarizer."),
    license = "?",
    keywords = "multi-document summarization",
    url = "http://",
    packages=['cuttsum'],
    long_description=read('README'),
    classifiers=[
#        "Development Status :: 3 - Alpha",
#        "Topic :: Utilities",
#        "License :: OSI Approved :: BSD License",
    ],
    include_package_data = True,
    package_data = setup_package_data()
        ## If any package contains *.txt or *.rst files, include them:
        #'': ['*.txt', '*.rst'],
        # And include any *.msg files found in the 'hello' package, too:
           # }
)
