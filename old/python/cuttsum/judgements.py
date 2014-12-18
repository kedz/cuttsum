class Match:
    def __init__(self, query_id, update_id, nugget_id, 
                 match_start, match_end, auto_p):
        self.query_id = query_id
        self.update_id = update_id
        self.nugget_id = nugget_id
        self.start = match_start
        self.end = match_end
        self.auto_p = auto_p

def read_matches_tsv(tsv, filter_query_id=None):
    matches = []
    with open(tsv, 'r') as f:
        f.readline()
        for line in f:
            items = line.strip().split('\t')
            query_id = int(items[0])
            update_id = items[1]
            nugget_id = items[2]
            match_start = int(items[3])
            match_end = int(items[4])
            auto_p = float(items[5])
            m = Match(query_id, update_id, nugget_id, 
                      match_start, match_end, auto_p)
            if filter_query_id is None or m.query_id == filter_query_id:
                matches.append(m)
    return matches

