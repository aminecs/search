from sentence_transformers import SentenceTransformer, util
import logging
import ast

DOCS_LEN = 500

def get_model(model_name):
    logging.info("LOADING MODEL: " + model_name)
    return SentenceTransformer(model_name)

def get_embeddings(model, doc):
    logging.info("CALCULATING EMBEDDINGS FOR DOC: " + str(doc["id"]))

    txt = ast.literal_eval(doc["text_processed"])
    txt_join = " ".join(txt)
    embedding = model.encode(txt_join)

    logging.info("EMBEDDINGS FOR DOC: " + str(doc["id"]) + " WAS CALCULATED")

    return embedding

def get_embeddings_for_docs(model, docs):
    logging.info("CALCULATING EMBEDDINGS FOR " + str(DOCS_LEN) + " DOCS")

    embeddings = []
    i = 0
    for doc in docs:
        logging.info(f"GENERATE EMBEDDING NUMBER {i}")
        embedding = get_embeddings(model, doc)
        embeddings.append([doc["id"], embedding])
        i += 1

    logging.info("EMBEDDINGS FOR " + str(DOCS_LEN) + " DOCS WAS CALCULATED")
    return embeddings