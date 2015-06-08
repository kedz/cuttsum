import cuttsum.events
import cuttsum.corpora
from cuttsum.pipeline import ArticlesResource
import pandas as pd
import datetime

def main():
    results = []
    res = ArticlesResource()

    for ext in ["gold", "goose"]:
        for event in cuttsum.events.get_2013_events():
            if event.query_id != "TS13.3":
                continue
            if event.query_id.startswith("TS13"):
                corpus = cuttsum.corpora.EnglishAndUnknown2013()
            else:
                raise Exception()
            min_hour = datetime.datetime(datetime.MAXYEAR, 1, 1)
            max_hour = datetime.datetime(datetime.MINYEAR, 1, 1)
            total = 0

            for hour, path, si in res.streamitem_iter(event, corpus, ext):
                if hour < min_hour:
                    min_hour = hour
                if hour > max_hour:
                    max_hour = hour
                total += 1
            results.append({"event": event.fs_name(),
                            "event start": event.list_event_hours()[0],
                            "event stop": event.list_event_hours()[-1],
                            "article start": min_hour,
                            "article stop": max_hour,
                            "total": total,
                            "annotator": ext})
    df = pd.DataFrame(results, 
        columns=["event", "annotator", "event start", "event stop",
                 "article start", "article stop", "total", "annotator"])
    print df

if __name__ == u"__main__":
    main()
