from haystack.components.routers import TransformersTextRouter

text_router = TransformersTextRouter(model="shahrukhx01/bert-mini-finetune-question-detection")
text_router.warm_up()

queries = [
    "Arya Stark father",  # Keyword Query
    "Who was the father of Arya Stark",  # Interrogative Query
    "Lord Eddard was the father of Arya Stark",  # Statement Query
]

result = text_router.run(text=queries[0])
next(iter(result))

import pandas as pd

results = {"Query": [], "Output Branch": [], "Class": []}

for query in queries:
    result = text_router.run(text=query)
    results["Query"].append(query)
    results["Output Branch"].append(next(iter(result)))
    results["Class"].append("Keyword Query" if next(iter(result)) == "LABEL_0" else "Question/Statement")

pd.DataFrame.from_dict(results)

text_router = TransformersTextRouter(model="shahrukhx01/question-vs-statement-classifier")
text_router.warm_up()

queries = [
    "Who was the father of Arya Stark",  # Interrogative Query
    "Lord Eddard was the father of Arya Stark",  # Statement Query
]

results = {"Query": [], "Output Branch": [], "Class": []}

for query in queries:
    result = text_router.run(text=query)
    results["Query"].append(query)
    results["Output Branch"].append(next(iter(result)))
    results["Class"].append("Question" if next(iter(result)) == "LABEL_1" else "Statement")

pd.DataFrame.from_dict(results)

text_router = TransformersTextRouter(model="cardiffnlp/twitter-roberta-base-sentiment")
text_router.warm_up()
queries = [
    "What's the answer?",  # neutral query
    "Would you be so lovely to tell me the answer?",  # positive query
    "Can you give me the damn right answer for once??",  # negative query
]
sent_results = {"Query": [], "Output Branch": [], "Class": []}

for query in queries:
    result = text_router.run(text=query)
    sent_results["Query"].append(query)
    sent_results["Output Branch"].append(next(iter(result)))
    sent_results["Class"].append({"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2":"positive"}.get(next(iter(result)), "Unknown"))

pd.DataFrame.from_dict(sent_results)

from haystack.components.routers import TransformersZeroShotTextRouter

text_router = TransformersZeroShotTextRouter(labels=["music", "cinema"])
text_router.warm_up()
queries = [
    "In which films does John Travolta appear?",  # cinema
    "What is the Rolling Stones first album?",  # music
    "Who was Sergio Leone?",  # cinema
]
sent_results = {"Query": [], "Output Branch": []}

for query in queries:
    result = text_router.run(text=query)
    sent_results["Query"].append(query)
    sent_results["Output Branch"].append(next(iter(result)))

pd.DataFrame.from_dict(sent_results)

from haystack.components.routers import TransformersZeroShotTextRouter

text_router = TransformersZeroShotTextRouter(labels=["Game of Thrones", "Star Wars", "Lord of the Rings"])
text_router.warm_up()

queries = [
    "Who was the father of Arya Stark",  # Game of Thrones
    "Who was the father of Luke Skywalker",  # Star Wars
    "Who was the father of Frodo Baggins",  # Lord of the Rings
]

results = {"Query": [], "Output Branch": []}

for query in queries:
    result = text_router.run(text=query)
    results["Query"].append(query)
    results["Output Branch"].append(next(iter(result)))

pd.DataFrame.from_dict(results)

from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()

from datasets import load_dataset
from haystack import Document

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

from haystack.components.embedders import SentenceTransformersDocumentEmbedder

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner

text_router = TransformersTextRouter(model="shahrukhx01/bert-mini-finetune-question-detection")
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
embedding_retriever = InMemoryEmbeddingRetriever(document_store)
bm25_retriever = InMemoryBM25Retriever(document_store)
document_joiner = DocumentJoiner()

from haystack import Pipeline

query_classification_pipeline = Pipeline()
query_classification_pipeline.add_component("text_router", text_router)
query_classification_pipeline.add_component("text_embedder", text_embedder)
query_classification_pipeline.add_component("embedding_retriever", embedding_retriever)
query_classification_pipeline.add_component("bm25_retriever", bm25_retriever)
query_classification_pipeline.add_component("document_joiner", document_joiner)

query_classification_pipeline.connect("text_router.LABEL_0", "text_embedder")
query_classification_pipeline.connect("text_embedder", "embedding_retriever")
query_classification_pipeline.connect("text_router.LABEL_1", "bm25_retriever")
query_classification_pipeline.connect("bm25_retriever", "document_joiner")
query_classification_pipeline.connect("embedding_retriever", "document_joiner")

# Useful for framing headers
equal_line = "=" * 30

# Run only the dense retriever on the full sentence query
res_1 = query_classification_pipeline.run({"text_router": {"text": "Who is the father of Arya Stark?"}})
print(f"\n\n{equal_line}\nQUESTION QUERY RESULTS\n{equal_line}")
print(res_1)

# Run only the sparse retriever on a keyword based query
res_2 = query_classification_pipeline.run({"text_router": {"text": "arya stark father"}})
print(f"\n\n{equal_line}\nKEYWORD QUERY RESULTS\n{equal_line}")
print(res_2)

from haystack.components.readers import ExtractiveReader

query_classification_pipeline = Pipeline()
query_classification_pipeline.add_component("bm25_retriever_0", InMemoryBM25Retriever(document_store))
query_classification_pipeline.add_component("bm25_retriever_1", InMemoryBM25Retriever(document_store))
query_classification_pipeline.add_component("text_router", TransformersTextRouter(model="shahrukhx01/question-vs-statement-classifier"))
query_classification_pipeline.add_component("reader", ExtractiveReader())

query_classification_pipeline.connect("text_router.LABEL_0", "bm25_retriever_0")
query_classification_pipeline.connect("bm25_retriever_0", "reader")
query_classification_pipeline.connect("text_router.LABEL_1", "bm25_retriever_1")

# Useful for framing headers
equal_line = "=" * 30

# Run the retriever + reader on the question query
query = "Who is the father of Arya Stark?"
res_1 = query_classification_pipeline.run({"text_router": {"text": query}, "reader": {"query": query}})
print(f"\n\n{equal_line}\nQUESTION QUERY RESULTS\n{equal_line}")
print(res_1)

# Run only the retriever on the statement query
query = "Arya Stark was the daughter of a Lord"
res_2 = query_classification_pipeline.run({"text_router": {"text": query}, "reader": {"query": query}})
print(f"\n\n{equal_line}\nKEYWORD QUERY RESULTS\n{equal_line}")
print(res_2)


