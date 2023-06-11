from google.cloud import bigquery
import tfidf


def get_tfidf_scores():
    client = bigquery.Client()

    query = """
        SELECT id, text
        FROM `cobalt-deck-389420.search_wiki_10000.wiki_10000`
        LIMIT 5
    """
    query_job = client.query(query)  # Make an API request.
    tf_idf_scores = tfidf.get_tfidf(query_job)
    

get_tfidf_scores()