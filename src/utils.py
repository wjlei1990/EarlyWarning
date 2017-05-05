import json


def load_json(fn):
    with open(fn) as fh:
        return json.load(fh)


def dump_json(content, fn):
    with open(fn, 'w') as fh:
        json.dump(content, fh, indent=2, sort_keys=True)
