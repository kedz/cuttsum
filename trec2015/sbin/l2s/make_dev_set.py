import cuttsum
import cuttsum.events
import random

random.seed(9780521686891)

events = [e for e in cuttsum.events.get_events() 
          if e.query_num not in [7, 24]]

random.shuffle(events)

for event in events[:5]:
    print event

#from collections import defaultdict
#t2e = defaultdict(list)
#for event in events:
#    t2e[event.type].append(event)
#
#for etype, typed_events in t2e.items():
#    print etype
#    for e in typed_events:
#        print "\t", e.title

