import collections, math
import logging

def get_tf(doc) -> dict:
    logging.info("CALCULATING TF FOR DOC: " + str(doc["id"])
                 
    txt = doc["text"]
    tf = collections.defaultdict(int)
    for word in txt:
        tf[word] += 1

    logging.info("TF FOR DOC: " + str(doc["id"]) + " WAS CALCULATED")

    return tf


def get_idf(docs : list) -> dict:
    logging.info("CALCULATING IDF FOR " + str(len(docs)) + " DOCS")

    doc_length = len(doc)
    idf = collections.defaultdict(int)
    for doc in docs:
        txt = doc["text"]
        for word in set(txt):
            idf[word] += 1
    for word in idf:
        idf[word] = math.log(doc_length/idf[word])

    logging.info("IDF WAS CALCULATED FOR " + str(len(docs)) + " DOCS")
    return idf

def get_tfidf(docs : list) -> dict:
    logging.info("CALCULATING TFIDF FOR " + str(len(docs)) + " DOCS")

    tfidf = collections.defaultdict(list)
    idf = get_idf(docs)

    for doc in docs:
        tf = get_tf(doc)
        doc_id = doc["id"]
        for word in idf:
            tfidf[word].append((doc_id,tf[word] * idf[word]))

    logging.info("TFIDF WAS CALCULATED FOR " + str(len(docs)) + " DOCS")
    return tfidf