from haystack.nodes import TransformersQueryClassifier

keyword_classifier = TransformersQueryClassifier()

queries = [
    "Arya Stark father",  # Keyword Query
    "Who was the father of Arya Stark",  # Interrogative Query
    "Lord Eddard was the father of Arya Stark",  # Statement Query
]

import pandas as pd

k_vs_qs_results = {"Query": [], "Output Branch": [], "Class": []}

for query in queries:
    result = keyword_classifier.run(query=query)
    k_vs_qs_results["Query"].append(query)
    k_vs_qs_results["Output Branch"].append(result[1])
    k_vs_qs_results["Class"].append("Question/Statement" if result[1] == "output_1" else "Keyword")

pd.DataFrame.from_dict(k_vs_qs_results)

question_classifier = TransformersQueryClassifier(model_name_or_path="shahrukhx01/question-vs-statement-classifier")

queries = [
    "Who was the father of Arya Stark",  # Interrogative Query
    "Lord Eddard was the father of Arya Stark",  # Statement Query
]

q_vs_s_results = {"Query": [], "Output Branch": [], "Class": []}

for query in queries:
    result = question_classifier.run(query=query)
    q_vs_s_results["Query"].append(query)
    q_vs_s_results["Output Branch"].append(result[1])
    q_vs_s_results["Class"].append("Question" if result[1] == "output_1" else "Statement")

pd.DataFrame.from_dict(q_vs_s_results)

from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http, convert_files_to_docs, clean_wiki_text

document_store = InMemoryDocumentStore(use_bm25=True)

doc_dir = "data/tutorial14"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt14.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

got_docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.write_documents(got_docs)

from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader, TransformersQueryClassifier

bm25_retriever = BM25Retriever(document_store=document_store)

embedding_retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
keyword_classifier = TransformersQueryClassifier()

from haystack.pipelines import Pipeline

transformer_keyword_classifier = Pipeline()
transformer_keyword_classifier.add_node(component=keyword_classifier, name="QueryClassifier", inputs=["Query"])
transformer_keyword_classifier.add_node(
    component=embedding_retriever, name="EmbeddingRetriever", inputs=["QueryClassifier.output_1"]
)
transformer_keyword_classifier.add_node(
    component=bm25_retriever, name="BM25Retriever", inputs=["QueryClassifier.output_2"]
)
transformer_keyword_classifier.add_node(
    component=reader, name="QAReader", inputs=["BM25Retriever", "EmbeddingRetriever"]
)

from haystack.utils import print_answers

# Useful for framing headers
equal_line = "=" * 30

# Run only the dense retriever on the full sentence query
res_1 = transformer_keyword_classifier.run(query="Who is the father of Arya Stark?")
print(f"\n\n{equal_line}\nQUESTION QUERY RESULTS\n{equal_line}")
print_answers(res_1, details="minimum")
print("\n\n")

# Run only the sparse retriever on a keyword based query
res_2 = transformer_keyword_classifier.run(query="arya stark father")
print(f"\n\n{equal_line}\nKEYWORD QUERY RESULTS\n{equal_line}")
print_answers(res_2, details="minimum")

question_classifier = TransformersQueryClassifier(model_name_or_path="shahrukhx01/question-vs-statement-classifier")

transformer_question_classifier = Pipeline()
transformer_question_classifier.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
transformer_question_classifier.add_node(
    component=question_classifier, name="QueryClassifier", inputs=["EmbeddingRetriever"]
)
transformer_question_classifier.add_node(component=reader, name="QAReader", inputs=["QueryClassifier.output_1"])

from haystack.utils import print_documents

# Useful for framing headers
equal_line = "=" * 30

# Run the retriever + reader on the question query
res_1 = transformer_question_classifier.run(query="Who is the father of Arya Stark?")
print(f"\n\n{equal_line}\nQUESTION QUERY RESULTS\n{equal_line}")
print_answers(res_1, details="minimum")
print("\n\n")

# Run only the retriever on the statement query
res_2 = transformer_question_classifier.run(query="Arya Stark was the daughter of a Lord.")
print(f"\n\n{equal_line}\nSTATEMENT QUERY RESULTS\n{equal_line}")
print_documents(res_2)

labels = ["LABEL_0", "LABEL_1", "LABEL_2"]

sentiment_query_classifier = TransformersQueryClassifier(
    model_name_or_path="cardiffnlp/twitter-roberta-base-sentiment",
    use_gpu=True,
    task="text-classification",
    labels=labels,
)
queries = [
    "What's the answer?",  # neutral query
    "Would you be so lovely to tell me the answer?",  # positive query
    "Can you give me the damn right answer for once??",  # negative query
]
import pandas as pd

sent_results = {"Query": [], "Output Branch": [], "Class": []}

for query in queries:
    result = sentiment_query_classifier.run(query=query)
    sent_results["Query"].append(query)
    sent_results["Output Branch"].append(result[1])
    if result[1] == "output_1":
        sent_results["Class"].append("negative")
    elif result[1] == "output_2":
        sent_results["Class"].append("neutral")
    elif result[1] == "output_3":
        sent_results["Class"].append("positive")

pd.DataFrame.from_dict(sent_results)

labels = ["music", "cinema"]

query_classifier = TransformersQueryClassifier(
    model_name_or_path="typeform/distilbert-base-uncased-mnli",
    use_gpu=True,
    task="zero-shot-classification",
    labels=labels,
)
queries = [
    "In which films does John Travolta appear?",  # cinema
    "What is the Rolling Stones first album?",  # music
    "Who was Sergio Leone?",  # cinema
]
import pandas as pd

query_classification_results = {"Query": [], "Output Branch": [], "Class": []}

for query in queries:
    result = query_classifier.run(query=query)
    query_classification_results["Query"].append(query)
    query_classification_results["Output Branch"].append(result[1])
    query_classification_results["Class"].append("music" if result[1] == "output_1" else "cinema")

pd.DataFrame.from_dict(query_classification_results)
