import streamcorpus as sc

def get_loc_sequences(doc):
    seqs = []
    for sentence in doc:
        buff = []
        for token in sentence:
            if token.ne == u'LOCATION':
                buff.append(token.surface)
            else:
                if len(buff) > 0:
                    seqs.append(u' '.join(buff))
                    buff = []
        if len(buff) > 0:
            seqs.append(u' '.join(buff))
            buff = []
    return seqs

def is_loc(token):
    netype = None
    if token.entity_type != 3 and token.entity_type != 4:
        netype = sc.get_entity_type(token)
    if netype == 'LOC':
        return True
    else:
        return False
