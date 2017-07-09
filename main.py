from collections import defaultdict
from copy import deepcopy
import json
import re

from bs4 import BeautifulSoup
import requests
from sklearn.externals import joblib


def main():
    text_clf = joblib.load('clf.pkl')

    with open('categories_mapping.json', 'r') as f:
       categories_mapping = json.load(f)

    with open('ads.json', 'r') as f:
       ads = json.load(f)

    inverse_categories_mapping = {v:k for k, v in categories_mapping.items()}
    ads_class_mapping = defaultdict(list)

    for ad in ads['ads']:
        ad_class = get_class_for_url(text_clf, ad['url'], description=ad['text'])

        print 'URL %s has class %s' % (ad['url'], inverse_categories_mapping[ad_class])

        ads_class_mapping[ad_class].append(ad)

    while True:
        url = raw_input('URL: ')
        probabilites = get_class_for_url(text_clf, url, proba=True)

        tuples = list(enumerate(probabilites))
        tuples.sort(key=lambda x: x[1], reverse=True)

        candidate1 = tuples[0]
        candidate2 = tuples[1]

        targeted_ads = deepcopy(ads_class_mapping[candidate1[0]])

        # Classes are near which means they are related.
        if candidate1[1] - candidate2[1] < 0.2:
            targeted_ads.extend(deepcopy(ads_class_mapping[candidate2[0]]))

        print targeted_ads


def get_class_for_url(text_clf, url, description='', proba=False):
    response = requests.get(url)

    if response.status_code != 200:
        print 'Unable to retrieve URL'

    soup = BeautifulSoup(response.content, 'html.parser')

    texts = soup.findAll(text=True)

    visible_texts = [text for text in texts if _visible(text)]

    if description:
        visible_texts.append(description)

    doc = ' '.join(visible_texts)

    if not proba:
        return text_clf.predict([doc])[0]

    return text_clf.predict_proba([doc])[0]


def _visible(element):
    """Method used to filter visible text elements. """
    if element.parent.name in ['style', 'script', '[document]', 'head', 'link']:
        return False
    elif re.match('<!--.*-->', element.encode('utf-8')):
        return False

    value = element.encode('utf-8')
    return not value.isspace()


if __name__ == '__main__':
    main()
