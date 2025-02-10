#RAGenerator和GenerativeQAPipeline在我当前版本中不可用。
#不能降版本，降了得大改，貌似有新的东西可以替代。
#这文件跑不了，只是记录一下。

import pandas as pd
from haystack.utils import fetch_archive_from_http

# Download sample
doc_dir = "data/tutorial7/"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/small_generator_dataset.csv.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Create dataframe with columns "title" and "text"
df = pd.read_csv(f"{doc_dir}/small_generator_dataset.csv", sep=",")
# Minimal cleaning
df.fillna(value="", inplace=True)

print(df.head())

from haystack import Document

# Use data to initialize Document objects
titles = list(df["title"].values)
texts = list(df["text"].values)
documents = []
for title, text in zip(titles, texts):
    documents.append(Document(content=text, meta={"name": title or ""}))

from haystack.document_stores import FAISSDocumentStore

document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

from haystack.nodes import RAGenerator, DensePassageRetriever
from haystack.nodes import PromptNode

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True,
    embed_title=True,
)

generator = RAGenerator(
    model_name_or_path="facebook/rag-token-nq",
    use_gpu=True,
    top_k=1,
    max_length=200,
    min_length=2,
    embed_title=True,
    num_beams=2,
)

# Delete existing documents in documents store
document_store.delete_documents()

# Write documents to document store
document_store.write_documents(documents)

# Add documents embeddings to index
document_store.update_embeddings(retriever=retriever)


from haystack.pipelines import GenerativeQAPipeline


pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)



from haystack.utils import print_answers

QUESTIONS = [
    "who got the first nobel prize in physics",
    "when is the next deadpool movie being released",
    "which mode is used for short wave broadcast service",
    "who is the owner of reading football club",
    "when is the next scandal episode coming out",
    "when is the last time the philadelphia won the superbowl",
    "what is the most current adobe flash player version",
    "how many episodes are there in dragon ball z",
    "what is the first step in the evolution of the eye",
    "where is gall bladder situated in human body",
    "what is the main mineral in lithium batteries",
    "who is the president of usa right now",
    "where do the greasers live in the outsiders",
    "panda is a national animal of which country",
    "what is the name of manchester united stadium",
]

for question in QUESTIONS:
    res = pipe.run(query=question, params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
    print_answers(res, details="minimum")