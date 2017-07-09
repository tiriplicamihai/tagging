import json

from bs4 import BeautifulSoup
import requests


URL = 'http://techtc.cs.technion.ac.il/techtc300/techtc300.html#software'
CATEGORIES_FILE = 'categories.json'

def main():
    """Parse the table from http://techtc.cs.technion.ac.il/techtc300/techtc300.html#software
    and extract for each ID the corresponding category.

    Output: A JSON file with a mapping between id and category.
    """
    response = requests.get(URL)

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')

    categories = {}
    # Ignore the table head.
    for row in table.find_all('tr')[1:]:
        # Each row has only two links which represent links to dmoz categories.
        for link in row.find_all('a'):
            href = link.attrs['href']
            # category is considered the first two levels of the URL path
            path = href.split('org')[-1][1:].split('/')
            categories[link.text] = '/'.join(path[:2])

    with open(CATEGORIES_FILE, 'w') as f:
        json.dump(categories, f)

if __name__ == '__main__':
    main()
