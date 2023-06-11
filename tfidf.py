import collections, math
import logging
import ast

DOCS_LEN = 1000

# TODO: Use TypedDict for type hinting of doc
def get_tf(doc) -> dict:
    logging.info("CALCULATING TF FOR DOC: " + str(doc["id"]))
                     
    txt = ast.literal_eval(doc["text_processed"])
    tf : dict = collections.defaultdict(int)
    for word in txt:
        tf[word] += 1

    logging.info("TF FOR DOC: " + str(doc["id"]) + " WAS CALCULATED")

    return tf


def get_idf(docs : list) -> dict:
    logging.info("CALCULATING IDF FOR " + str(DOCS_LEN) + " DOCS")

    idf : dict = collections.defaultdict(int)
    i = 1
    for doc in docs:
        logging.info(f"GETTING IDF FOR DOC NUMBER {i}")
        txt = ast.literal_eval(doc["text_processed"])
        for word in set(txt):
            idf[word] += 1
        i += 1

    for word in idf:
        idf[word] = math.log(DOCS_LEN/idf[word])

    logging.info("IDF WAS CALCULATED FOR " + str(DOCS_LEN) + " DOCS")
    return idf

def get_tfidf(docs : list) -> dict:
    logging.info("CALCULATING TFIDF FOR " + str(DOCS_LEN) + " DOCS")

    tfidf = []
    idf = get_idf(docs)

    i = 1
    for doc in docs:
        logging.info(f"GETTING TFIDF FOR DOC NUMBER {i}")
        tf = get_tf(doc)
        doc_id = doc["id"]
        for word in idf:
            tfidf.append((word, doc_id, tf[word] * idf[word]))
        i += 1

    logging.info("TFIDF WAS CALCULATED FOR " + str(DOCS_LEN) + " DOCS")
    return tfidf