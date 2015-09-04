TREC_HOME=`pwd`
TREC_DATA=$TREC_HOME/trec-data
# Replace these with your paths
JAVA_HOME=$TREC_HOME/jdk1.8.0_25
CORENLP_DIR=/home/kedz/javalibs/stanford-corenlp-full-2015-04-20
CORENLP_VER=3.5.2

virtualenv env
echo "export JAVA_HOME=$JAVA_HOME" >> env/bin/activate
echo "export CORENLP_DIR=$CORENLP_DIR" >> env/bin/activate
echo "export CORENLP_VER=$CORENLP_VER" >> env/bin/activate
echo "export TREC_HOME=$TREC_HOME" >> env/bin/activate
echo "export TREC_DATA=$TREC_DATA" >> env/bin/activate
echo "export SRILM=$TREC_HOME/srilm" >> env/bin/activate
echo "export SRILM_INC=\$SRILM/include" >> env/bin/activate
echo "export SRILM_LIB=\$SRILM/lib/i686-m64" >> env/bin/activate
echo "export LD_LIBRARY_PATH=\$TREC_HOME/local/lib/:\$TREC_HOME/openmpi/lib/:\$LD_LIBRARY_PATH" >> env/bin/activate
echo "export MPICC=\$TREC_HOME/openmpi/bin/mpicc" >> env/bin/activate
echo "export PATH=\$TREC_HOME/openmpi/bin/:\$SRILM/bin/:\$SRILM/bin/i686-m64/:\$JAVA_HOME/jdk1.8.0_25/bin:\$TREC_HOME/local/bin/:\$PATH" >> env/bin/activate
echo "export OMP_THREAD_LIMIT=1" >> env/bin/activate
echo "export OMP_NUM_THREADS=1" >> env/bin/activate
echo "export PYTHONPATH=\$PYTHONPATH:\$TREC_HOME/vowpal_wabbit/python" >> env/bin/activate

source env/bin/activate

git clone https://github.com/kedz/corenlp.git
cd corenlp
python setup.py develop
cd ..

pip install -U setuptools
pip install numpy 
pip install scipy
pip install cython

pip install pandas

git clone https://github.com/kedz/sumpy.git
cd sumpy
python setup.py develop
cd ..




mkdir $SRILM
tar zxvf $TREC_HOME/srilm-1.7.1.tar.gz -C $SRILM
cd $SRILM
MAKE_PIC=yes make World
cd $TREC_HOME

wget http://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.5.tar.gz
tar zxvf openmpi-1.8.5.tar.gz
cd openmpi-1.8.5
./configure --prefix=$TREC_HOME/openmpi
make
make install
cd $TREC_HOME

git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py
python setup.py install
cd $TREC_HOME

git clone https://github.com/kedz/wtmf.git
cd wtmf 
python setup.py install
cd $TREC_HOME

git clone https://github.com/JohnLangford/vowpal_wabbit.git
cd vowpal_wabbit
make
cd python
make
cd $TREC_HOME 

pip install regex

#tar zxvf libxml2-2.9.1.tar.gz
#cd libxml2-2.9.1
#./configure --prefix=$TREC_HOME/local
#make
#make install
#cd $TREC_HOME
#
#tar zxvf libxslt-1.1.28.tar.gz
#cd libxslt-1.1.28
#./configure --prefix=$TREC_HOME/local
#make
#make install
#cd $TREC_HOME

#git clone https://github.com/lxml/lxml.git
#cd lxml
#python setup.py build_ext -I $TREC_HOME/local/include/libxml2:$TREC_HOME/local/include/ -L $TREC_HOME/local/lib
#python setup.py install
#cd $TREC_HOME 
pip install lxml


pip install pandas 
pip install beautifulsoup4
pip install pillow

git clone https://github.com/kedz/cuttsum.git
cd cuttsum/trec2015
python setup.py develop


