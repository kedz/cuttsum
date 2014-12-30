import os
import inspect
from setuptools import setup, Extension

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def setup_package_data():
    """
Set up package data -- these are mostly resources generated from TS track
organizers."""

            
    data = [
            # 2013 test event xml
            os.path.join(u'2013-data', u'trec2013-ts-topics-test.xml'),
            
            # 2013 nugget and match data, with some query id correction
            # by me (Chris Kedzie). 
            os.path.join(u'2013-data', u'nuggets.tsv.gz'),
            os.path.join(u'2013-data', u'matches.tsv.gz'),
           
            # 2014 test event xml
            os.path.join(u'2014-data', u'trec2014-ts-topics-test.xml'),
            
            # 2014 nugget and match data, untouched by me.
            os.path.join(u'2014-data', u'nuggets.tsv.gz'),
            os.path.join(u'2014-data', u'matches.tsv.gz'),

            # Pooled updates from 2014 track participants.
            os.path.join(u'2014-data', u'updates_sampled.tsv.gz'),
            os.path.join(u'2014-data', u'updates_sampled.extended.tsv.gz'),
            os.path.join(u'2014-data', u'updates_sampled.levenshtein.tsv.gz'),
            
            # Complete set of 2014 updates from our official run submissions. 
            os.path.join(u'2014-data', u'cunlp-updates.ssv.gz'),

            # Article Classifier Model.
            os.path.join(u'models'),
            os.path.join(u'models', u'article_clf.pkl'),      
            os.path.join(u'models', u'article_clf.pkl_01.npy'),
            os.path.join(u'models', u'article_clf.pkl_02.npy'),
            os.path.join(u'models', u'article_clf.pkl_03.npy'),
            os.path.join(u'models', u'article_clf.pkl_04.npy'),
            os.path.join(u'models', u'article_clf.pkl_05.npy'),
            os.path.join(u'models', u'article_vectorizer.pkl'),]
         
    return {'cuttsum': data}
    
def build_srilm_extension():

    try:
        from Cython.Build import cythonize
        use_cython = True
        ext = ".pyx"
    except ImportError, e:
        use_cython = False
        ext = ".cpp"

    sources = [os.path.join('cuttsum', 'srilm{}'.format(ext))]
    libraries = ['oolm', 'dstruct', 'misc', 'z', 'gomp']

    srilm_inc = os.getenv(u"SRILM_INC", None)
    if srilm_inc is None:
        print "Please set SRILM_INC to the location of SRILM" \
            " `include' directory"
        import sys
        sys.exit()

    srilm_lib = os.getenv(u"SRILM_LIB", None)
    if srilm_lib is None:
        print "Please set SRILM_LIB to the location of SRILM" \
            " `lib' directory"
        import sys
        sys.exit()

    extension = Extension(
        "cuttsum.srilm",
        sources=sources,
        libraries=libraries,
        extra_compile_args=["-I{}".format(srilm_inc)],
        extra_link_args=["-L{}".format(srilm_lib)],
        language="c++")

    if use_cython is True:
        return cythonize(extension)
    else:
        return extension



#extension = Extension(

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
    include_package_data=True,
    package_data=setup_package_data(),
    ext_modules=build_srilm_extension(),
)
