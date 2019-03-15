"""Get synonyms of the given list of words with Python package vocabulary

Usage: python expand_keywords.py query_filename expand_query_filename
"""
# TODO remove out-of-scope keywords


import json
import sys

from nltk.corpus import wordnet as wn
from vocabulary.vocabulary import Vocabulary as vb
from word_forms.word_forms import get_word_forms


# https: // github.com/tasdikrahman/vocabulary
# http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
# https://stevenloria.com/wordnet-tutorial/

def get_synonyms(word, method='vocabulary'):
    if method == 'vocabulary':
        ss = vb.synonym(word)
        if not ss:
            return []
        return [i['text'] for i in json.loads(ss)]
    elif method == 'wordnet':
        return [l.name() for s in wn.synsets(word) for l in s.lemmas()]


def get_all_forms(word):
    forms = []
    for values in get_word_forms(word).values():
        forms.extend(values)
    return forms


def main():
    filename = sys.argv[1]
    with open(filename) as file:
        line = file.readline().strip()
    words = line.split('"')
    words = [w for w in words if w != ' ' and w != '']
    # print(words)

    for word in words:
        res = []
        if ' ' not in word:
            all_forms = get_all_forms(word.lower())
        else:
            res.append(word)
            all_forms = []
        if word not in all_forms:
            all_forms.append(word)
        for word_form in all_forms:
            # print(word_form)

            res.extend(get_synonyms(word_form, 'vocabulary'))
            res.extend(get_synonyms(word_form, 'wordnet'))
            # res.extend(
            #     [l.name() for s in wn.synsets(word) for h in s.hypernyms() for l in
            #      h.lemmas()])
            # res.extend(
            #     [l.name() for s in wn.synsets(word) for h in s.hyponyms() for l in
            #      h.lemmas()])
            # res.extend(
            #         [l.name() for s in wn.synsets(word) for h in s.member_holonyms() for
            #          l in h.lemmas()])
            # res.extend(
            #     [l.name() for s in wn.synsets(word) for h in s.part_meronyms() for
            #      l in h.lemmas()])

        res = set(res)

        res_forms = []
        for r in res:
            res_forms.append(r)
            if ' ' not in r:
                res_forms.extend(get_all_forms(r.lower()))

        res = set(res_forms)
        for i in res:
            print(i.replace('_', ' '))


if __name__ == '__main__':
    main()
