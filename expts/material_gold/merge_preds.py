import json
import os
import sys

from docutils.io import InputError
from material_constants import PRED_DIRS_sent
from material_constants import PRED_DIRS

pred_dir_prefix = os.path.join('predictions')


def main():
    global sent
    if sys.argv[1] == 'sent':
        sent = True
        dirs = PRED_DIRS_sent
    elif sys.argv[1] == 'doc':
        sent = False
        dirs = PRED_DIRS
    else:
        raise InputError('Usage: python merge_preds.py sent/doc')

    for basedir in dirs:
        for subdir in dirs[basedir]:
            if not basedir.startswith(sys.argv[1]):
                continue
            dir = os.path.join(pred_dir_prefix, basedir, subdir)
            if not os.path.exists(dir):
                continue
            print(dir)
            for arch in os.listdir(dir):
                pred_dir = os.path.join(dir, arch)
                if not os.path.isdir(pred_dir):
                    continue
                print(pred_dir)
                if not os.path.exists(pred_dir):
                    continue
                merge_predictions(pred_dir)


def merge_predictions(pred_dir):
    global sent
    merged_preds = {}
    for filename in os.listdir(pred_dir):
        if not filename.endswith('json') or filename.startswith('merged'):
            continue
        domain = filename[:filename.find('.json')]
        print(os.path.join(pred_dir, filename))
        with open(os.path.join(pred_dir, filename), 'rt') as file:
            data = json.load(file, encoding='utf-8')
        if sent:
            # TODO
            for item in data:
                id = item['id']
                doc_id = id[:id.rfind('_')]
                sent_id = id[id.rfind('_') + 1:]
                if doc_id not in merged_preds:
                    merged_preds[doc_id] = {}
                if sent_id not in merged_preds[doc_id]:
                    merged_preds[doc_id][sent_id] = {}
                merged_preds[doc_id][sent_id][domain] = item['1']

        else:
            for item in data:
                doc_id = item['id']
                if doc_id not in merged_preds:
                    merged_preds[doc_id] = {}
                merged_preds[doc_id][domain] = item['1']
    with open(os.path.join(pred_dir, 'merged.json'), 'wt') as file:
        json.dump(merged_preds, file, ensure_ascii=False)


if __name__ == '__main__':
    main()
