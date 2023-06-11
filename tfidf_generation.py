from google.cloud import bigquery, storage
import tfidf
import os, sys
import pandas as pd
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/home/fromamine/.config/gcloud/application_default_credentials.json"


def get_tfidf_scores():
    logging.info("INIT BIGQUERY CLIENT")
    client = bigquery.Client()

    query = """
        SELECT id, text_processed
        FROM `cobalt-deck-389420.search_wiki_10000.wiki_10000`
    """

    logging.info("QUERY TO RETRIEVE DOCS FORMATTED")

    query_job = client.query(query)

    logging.info("QUERY PERFORMED")

    tf_idf_scores = tfidf.get_tfidf(query_job)

    logging.info("GET TF-IDF SCORES")
    logging.info("BUILDING DF")
    df = pd.DataFrame(tf_idf_scores, columns=['word','id','score'])

    logging.info("DF READY")

    logging.info("INIT CLOUD STORAGE")
    client_storage = storage.Client()
    bucket = client_storage.get_bucket("fromamine-search-bucket")
    logging.info("CLOUD STORAGE BUCKET OBTAINED")

    bucket.blob("datasets/tf_idf_wiki_10000.csv").upload_from_string(df.to_csv(), "text/csv")
    logging.info("DATA UPLOADED")

    

get_tfidf_scores()