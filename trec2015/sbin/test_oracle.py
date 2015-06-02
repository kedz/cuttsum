import cuttsum.summarizers
import cuttsum.events
import cuttsum.corpora


event = cuttsum.events.get_events(by_query_ids=["TS13.1"])[0]
corpus = cuttsum.corpora.EnglishAndUnknown2013()

print event


resource = cuttsum.summarizers.MonotoneSubmodularOracle()
resource.do_job_unit(event, corpus, 0)

