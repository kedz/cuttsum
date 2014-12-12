import streamcorpus as sc

def get_loc_sequences(sentence):
    seqs = []
    buff = []
    for token in sentence.tokens:
        if is_loc(token):
            buff.append(token.token.decode('utf-8'))
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
