from datetime import datetime

class Nugget:
    def __init__(self, query_id, nugget_id, timestamp, importance,
                 length, text):
        self.query_id = query_id
        self.nugget_id = nugget_id
        self.timestamp = timestamp
        self.importance = importance
        self.length = length
        self.text = text

def read_nuggets_tsv(tsv, filter_query_id=None):
    nuggets = []
    with open(tsv, 'r') as f:
        f.readline()
        for line in f:
            items = line.strip().split('\t')
            query_id = items[0]
            nugget_id = items[1]
            timestamp = datetime.utcfromtimestamp(int(items[2]))
            importance = int(items[3])
            length = int(items[4])
            text = items[5]
            nugget = Nugget(query_id, nugget_id, timestamp,
                            importance, length, text)
            if filter_query_id is None or nugget.query_id == filter_query_id:
                nuggets.append(nugget)
    return nuggets
