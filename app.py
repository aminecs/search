from nltk.tokenize import word_tokenize
import nltk
from google.cloud import bigquery
import os
import collections, ast, re
from sklearn.metrics.pairwise import cosine_similarity
import embeddings
import openai
import numpy as np
import anthropic

#nltk.download('all-nltk')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/home/fromamine/.config/gcloud/application_default_credentials.json"

client = bigquery.Client()

def get_query_preprocessed(query):
    return word_tokenize(query.lower())

def get_similar_docs(query_preprocessed : list) -> list:
    query_preproccesed_str = '"'+'","'.join(query_preprocessed)+'"'


    query = f"""
    SELECT id, sum(score) as total_score 
    FROM `cobalt-deck-389420.search_wiki_10000.tfidf_wiki_10000`
    WHERE word IN ({query_preproccesed_str})
    GROUP BY id
    HAVING total_score > 0
    ORDER BY total_score desc
    """
        
    query_job = client.query(query)
    
    return query_job

def get_relevant_docs_embeddings(docs):
    docs_ids = []

    for doc in docs:
        docs_ids.append(doc["id"])


    docs_ids_str = ", ".join(str(doc_id) for doc_id in docs_ids)

    query = f"""
    SELECT id, embedding
    FROM `cobalt-deck-389420.search_wiki_10000.embeddings_wiki_10000`
    WHERE id in ({docs_ids_str})
    """
        
    query_job = client.query(query)

    return query_job

def get_filtered_relevant_docs(docs, query):
    model = embeddings.get_model("multi-qa-mpnet-base-dot-v1")

    txt_join = " ".join(query)
    embeddings_query = model.encode(txt_join)

    cosine_similarity_scores = []

    for doc in docs:
        doc_embedding = doc["embedding"].strip('[]')
        doc_embedding = [float(token) for token in doc_embedding.split()]

        doc_embedding = np.array(doc_embedding)
        
        cosine_similarity_score = cosine_similarity(embeddings_query.reshape(1, -1), doc_embedding.reshape(1, -1))
        if cosine_similarity_score > 0.4:
            cosine_similarity_scores.append((cosine_similarity_score[0][0], doc["id"]))
    
    cosine_similarity_scores.sort(reverse=True)
    return cosine_similarity_scores

def get_docs_data(docs_scored):

    docs_ids_str = ", ".join(str(doc_id) for _, doc_id in docs_scored)

    query = f"""
        SELECT id, url, title, text
        FROM `cobalt-deck-389420.search_wiki_10000.wiki_10000`
        WHERE id in ({docs_ids_str})
    """

    query_job = client.query(query)

    docs_data_dict_list = []
    i = 1
    for row in query_job:
        doc_data_dict = {"id": i, "url": row["url"], "text": row["text"]}
        docs_data_dict_list.append(doc_data_dict)
        i += 1
    return docs_data_dict_list

def getClaudeApiKey():
    return os.environ.get('CLAUDE_API_KEY')

def getClient():
    api_key = getClaudeApiKey()
    return anthropic.Client(api_key)

def get_llm_answer(docs_scored, query):
    print("Number of docs considered: ", len(docs_scored))

    prompt = f"""
    Based on the following list of json documents and ONLY this list: {docs_scored},
    Answer the following question: {{ {query} }}
    You have to answer the question, and give the id and url of the text that has the answer.
    You also need to output the number of documents you had access to to find the answer.
    """

    client = getClient()
    payload = prompt
    response = client.completion(
    prompt=f"{anthropic.HUMAN_PROMPT}{payload}?{anthropic.AI_PROMPT}",
    stop_sequences = [anthropic.HUMAN_PROMPT],
    model="claude-v1",
    max_tokens_to_sample=100,
    )
    return response["completion"]
    



def get_results(query):
    query_preprocessed = get_query_preprocessed(query)
    pre_filtered_docs = get_similar_docs(query_preprocessed)
    relevant_docs_embeddings = get_relevant_docs_embeddings(pre_filtered_docs)
    relevant_docs_scored = get_filtered_relevant_docs(relevant_docs_embeddings, query_preprocessed)
    docs_data_dict_list = get_docs_data(relevant_docs_scored)
    answer = get_llm_answer(docs_data_dict_list, query)
    print(answer)


get_results("In which book of the bible can we find Abel?")