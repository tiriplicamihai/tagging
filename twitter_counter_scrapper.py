import json
import sys

import requests
from bs4 import BeautifulSoup

def main(url):
    response = requests.get(url)
    if response.status_code != 200:
        print "Failed to read url"
        sys.exit(1)

    soup = BeautifulSoup(response.content, 'html.parser')

    handlers = []
    for anchor in soup.findAll('a', {'class': 'uname'}):
        handlers.append(anchor.text)

    file_name = '%s.json' % url.split('/')[-1]
    with open(file_name.replace('-', '_'), 'w') as f:
        json.dump({'handlers': handlers}, f)

if __name__ == '__main__':
    main(sys.argv[1])
