install: wrapper
	mv *.o *.so lib

wrapper: SRILMClient.o SRILMClientWrapper.o
	g++  -shared -Wall -pthread -L$(BOOST_LIB) -L$(SRILM_LIB)   SRILMClientWrapper.o SRILMClient.o -loolm -ldstruct -lmisc -lboost_python  -fpic -o srilm_client.so

SRILMClientWrapper.o: SRILMClientWrapper.cc
	g++ -Wall -c SRILMClientWrapper.cc -I$(SRILM_INC) -I$(PY_INC) -fpic

SRILMClient.o: SRILMClient.cc
	g++ -Wall -c SRILMClient.cc -I$(SRILM_INC) -fpic

clean:
	rm -f test *.o *.so
