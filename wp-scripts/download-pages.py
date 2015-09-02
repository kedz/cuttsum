import gzip
import os
import re
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings()
import json

#list_path = "wp-lists/accident.tsv.gz"
#path = "wp-pages/accidents"


def main(input_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    url = 'https://en.wikipedia.org/w/api.php'
    http = urllib3.connection_from_url(url)
    page_url_tmp = 'https://en.wikipedia.org/w/index.php?title={}&oldid={}'

    queue = set()
    with gzip.open(input_path, "r") as f:
        f.readline()
        for line in f:
            queue.add(line.strip().split("\t")[3])

    for i, title in enumerate(queue, 1):
        #if i < 9950: continue 
        fields = {'action':'query', 'format':'json', 'prop':'revisions',
                  'titles':title, 'rvprop':'timestamp|ids',
                  'rvstart':'20111001000000', 'rvdir':'older',
                  'continue': ''}
        print title, i, "/", len(queue)
        r = http.request_encode_url('GET', url, fields=fields)
        result = json.loads(r.data)
        pid = result['query']['pages'].keys()[0]
        revs = result['query']['pages'][pid].get('revisions', None)
        if revs is None:
            #result_queue.put([])
            continue

        if len(revs) == 0:
            #result_queue.put([])
            continue
        rev = revs[0]

        page_url = page_url_tmp.format(
            title.replace(' ', '_'), rev['parentid'])
        html_content = http.request_encode_url('GET', page_url).data

        results = []

        soup = BeautifulSoup(html_content)
        div = soup.find("div", {"id": "mw-content-text"})

        if div is None:
            continue
        ofile = os.path.join(
            output_path, 
            title.replace(" ", "_").replace("/", "-") + ".txt")
        with open(ofile, "w") as f:
            for tag in div.find_all(True, recursive=False):
                if tag.name == 'p':
                    text = tag.get_text()
                    text = re.sub(r'\[\d+\]', '', tag.get_text())
                    if isinstance(text, unicode):
                        text = text.encode("utf-8")
                    f.write(text + "\n")
                    #text = cnlp.annotate(text)


if __name__ == u"__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args.input, args.output)

