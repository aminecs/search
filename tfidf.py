import collections, math
import logging

DOCS_LEN = 10000

# TODO: Use TypedDict for type hinting of doc
def get_tf(doc) -> dict:
    logging.info("CALCULATING TF FOR DOC: " + str(doc["id"]))
                     
    txt = doc["text_processed"]
    tf : dict = collections.defaultdict(int)
    for word in txt:
        tf[word] += 1

    logging.info("TF FOR DOC: " + str(doc["id"]) + " WAS CALCULATED")

    return tf


def get_idf(docs : list) -> dict:
    logging.info("CALCULATING IDF FOR " + str(DOCS_LEN) + " DOCS")

    idf : dict = collections.defaultdict(int)
    for doc in docs:
        txt = doc["text_processed"]
        for word in set(txt):
            idf[word] += 1
    for word in idf:
        idf[word] = math.log(DOCS_LEN/idf[word])

    logging.info("IDF WAS CALCULATED FOR " + str(DOCS_LEN) + " DOCS")
    return idf

def get_tfidf(docs : list) -> dict:
    logging.info("CALCULATING TFIDF FOR " + str(DOCS_LEN) + " DOCS")

    tfidf = []
    idf = get_idf(docs)

    for doc in docs:
        tf = get_tf(doc)
        doc_id = doc["id"]
        for word in idf:
            tfidf.append((word, doc_id, tf[word] * idf[word]))

    logging.info("TFIDF WAS CALCULATED FOR " + str(DOCS_LEN) + " DOCS")
    return tfidf