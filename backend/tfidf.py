import collections

def get_tf(doc):
    tf = collections.defaultdict(int)
    for word in doc:
        tf[word] += 1
    return tf