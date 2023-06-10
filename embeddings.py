from sentence_transformers import SentenceTransformer, util
import logging

def get_model(model_name):
    logging.info("LOADING MODEL: " + model_name)
    return SentenceTransformer(model_name)

def get_embeddings(model, doc):
    logging.info("CALCULATING EMBEDDINGS FOR DOC: " + str(doc["id"]))

    txt = doc["text"]
    embedding = model.encode(txt)

    logging.info("EMBEDDINGS FOR DOC: " + str(doc["id"]) + " WAS CALCULATED")

    return embedding

def get_embeddings_for_docs(model, docs):
    logging.info("CALCULATING EMBEDDINGS FOR " + str(len(docs)) + " DOCS")

    embeddings = dict()
    for doc in docs:
        embeddings[doc["id"]] = get_embeddings(model, doc)

    logging.info("EMBEDDINGS FOR " + str(len(docs)) + " DOCS WAS CALCULATED")
    return embeddings