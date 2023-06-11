from sentence_transformers import SentenceTransformer, util
import logging

DOCS_LEN = 10000

def get_model(model_name):
    logging.info("LOADING MODEL: " + model_name)
    return SentenceTransformer(model_name)

def get_embeddings(model, doc):
    logging.info("CALCULATING EMBEDDINGS FOR DOC: " + str(doc["id"]))

    txt = doc["text_processed"]
    embedding = model.encode(txt)

    logging.info("EMBEDDINGS FOR DOC: " + str(doc["id"]) + " WAS CALCULATED")

    return embedding

def get_embeddings_for_docs(model, docs):
    logging.info("CALCULATING EMBEDDINGS FOR " + str(DOCS_LEN) + " DOCS")

    embeddings = []
    for doc in docs:
        embedding = get_embeddings(model, doc)
        embeddings.append([doc["id"], embedding])

    logging.info("EMBEDDINGS FOR " + str(DOCS_LEN) + " DOCS WAS CALCULATED")
    return embeddings