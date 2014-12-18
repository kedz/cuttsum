#include <boost/python.hpp>
#include "SRILMClient.h"
using namespace boost::python;

BOOST_PYTHON_MODULE(srilm_client)
{
    class_<SRILMClient>("Client")
        .def(init<char*, bool>())
        .def("sentence_log_prob", &SRILMClient::sentenceLogProb);
}
