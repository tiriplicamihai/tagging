import json
import sys

import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://www.socialbakers.com'

def main(url, category):
    response = requests.get(url)
    if response.status_code != 200:
        print "Failed to read url"
        sys.exit(1)

    soup = BeautifulSoup(response.content, 'html.parser')

    links = []
    for anchor in soup.findAll('a', {'class': 'acc-placeholder-img'}):
        links.append(anchor['href'])

    pages = []
    for link in links:
        response = requests.get(BASE_URL + link)
        if response.status_code != 200:
            print "Failed to read link %s" % link
            continue
        soup = BeautifulSoup(response.content, 'html.parser')
        pages.append(soup.findAll('a', {'class': 'blank show-tooltip'})[0]['href'])


    with open('%s_pages.json' % category, 'w') as f:
        json.dump({'pages': list(set(pages))}, f)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
