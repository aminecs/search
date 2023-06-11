from datasets import load_dataset
import pandas as pd
from google.cloud import bigquery, storage
from google import auth
import os
from nltk.tokenize import word_tokenize
import nltk

client = bigquery.Client()
nltk.download('all-nltk')

'''
Wikipedia dataset

features: ['id', 'url', 'title', 'text'],
num_rows: 6458670
'''

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/home/fromamine/.config/gcloud/application_default_credentials.json"

dataset = load_dataset("wikipedia", "20220301.en", beam_runner="DirectRunner")
df = pd.DataFrame.from_dict(dataset['train'][:10000]) # Get 10000 rows
df["text_processed"] = df["text"].apply(lambda x: word_tokenize(x.lower()))
client = storage.Client()
bucket = client.get_bucket("fromamine-search-bucket")
    
bucket.blob("datasets/wiki_10000.csv").upload_from_string(df.to_csv(), "text/csv")