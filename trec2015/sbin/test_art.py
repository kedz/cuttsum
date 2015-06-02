import cuttsum.pipeline
import cuttsum.events
import cuttsum.corpora


event = cuttsum.events.get_events(by_query_ids=["TS13.1"])[0]
corpus = cuttsum.corpora.EnglishAndUnknown2013()

print event


resource = cuttsum.pipeline.ArticlesResource()
units = resource.get_job_units(event, corpus)

print units

for unit in units:
    print unit
    resource.do_job_unit(event, corpus, unit)
