import collections, math

def get_tf(doc) -> dict:
    txt = doc["text"]
    tf = collections.defaultdict(int)
    for word in txt:
        tf[word] += 1
    return tf


def get_idf(docs : list) -> dict:
    doc_length = len(doc)
    idf = collections.defaultdict(int)
    for doc in docs:
        txt = doc["text"]
        for word in set(txt):
            idf[word] += 1
    for word in idf:
        idf[word] = math.log(doc_length/idf[word])
    return idf

def get_tfidf(docs : list) -> dict:
    tfidf = collections.defaultdict(list)
    idf = get_idf(docs)

    for doc in docs:
        tf = get_tf(doc)
        doc_id = doc["id"]
        for word in idf:
            tfidf[word].append((doc_id,tf[word] * idf[word]))

    return tfidf


