from haystack.utils import fetch_archive_from_http
import os
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import PreProcessor
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import EvaluationResult,MultiLabel




# Download evaluation data, which is a subset of Natural Questions development set containing 50 documents with one question per document and multiple annotated answers
doc_dir = "data/tutorial5"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v2.json.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

doc_index = "tutorial5_docs"
label_index = "tutorial5_labels"

host = os.environ.get("ELASTICSEAECH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(host=host,
                                            username="",
                                            password="",
                                            index=doc_index,
                                            label_index=label_index,
                                            embedding_field="emb",
                                            embedding_dim=768,
                                            excluded_meta_data=["emb"],)

preprocessor = PreProcessor(split_by="word",
                            split_length=200,
                            split_overlap=0,
                            split_respect_sentence_boundary=False,
                            clean_empty_lines=False,
                            clean_whitespace=False,)
document_store.delete_all_documents(index=doc_index)
document_store.delete_all_documents(index=label_index)
document_store.add_eval_data(filename="data/tutorial5/nq_dev_subset_v2.json",
                             doc_index=doc_index,
                             label_index=label_index,
                             preprocessor=preprocessor,)

retriever = BM25Retriever(document_store=document_store)
reader = FARMReader("deepset/roberta-base-squad2",
                    top_k=4,
                    return_no_answer=True,)
pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)


eval_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True,
                                                       drop_no_answers=True,)
eval_result = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": 5}})

retriever_result = eval_result["Retriever"]
retriever_result.head()

reader_result = eval_result["Reader"]
reader_result.head()

query = "who is written in the book of life"
retriever_book_of_life = retriever_result[retriever_result["query"] == query]

reader_book_of_life = reader_result[reader_result["query"] == query]

eval_result.save("../")

saved_eval_result = EvaluationResult.load("../")
metrics = saved_eval_result.calculate_metrics()
print(f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
print(f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')

print(f'Reader - F1-Score: {metrics["Reader"]["f1"]}')
print(f'Reader - Exact Match: {metrics["Reader"]["exact_match"]}')

pipeline.print_eval_report(saved_eval_result)

advanced_eval_result = pipeline.eval(
    labels=eval_labels, params={"Retriever": {"top_k": 5}}, sas_model_name_or_path="cross-encoder/stsb-roberta-large"
)

metrics = advanced_eval_result.calculate_metrics()
print(metrics["Reader"]["sas"])

eval_result_with_upper_bounds = pipeline.eval(
    labels=eval_labels, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 5}}, add_isolated_node_eval=True
)

pipeline.print_eval_report(eval_result_with_upper_bounds)

metrics = saved_eval_result.calculate_metrics(answer_scope="context")
print(f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
print(f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')

print(f'Reader - F1-Score: {metrics["Reader"]["f1"]}')
print(f'Reader - Exact Match: {metrics["Reader"]["exact_match"]}')
document_store.get_all_documents()[0]
# Let's try Document Retrieval on a file level (it's sufficient if the correct file identified by its name (for example, 'Book of Life') was retrieved).
eval_result_custom_doc_id = pipeline.eval(
    labels=eval_labels, params={"Retriever": {"top_k": 5}}, custom_document_id_field="name"
)
metrics = eval_result_custom_doc_id.calculate_metrics(document_scope="document_id")
print(f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
print(f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')
# Let's enforce the context again:
metrics = eval_result_custom_doc_id.calculate_metrics(document_scope="document_id_and_context")
print(f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
print(f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')

import tempfile
from pathlib import Path
from haystack.nodes import PreProcessor
from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore()

label_preprocessor = PreProcessor(
    split_length=200,
    split_overlap=0,
    split_respect_sentence_boundary=False,
    clean_empty_lines=False,
    clean_whitespace=False,
)

# The add_eval_data() method converts the given dataset in json format into Haystack document and label objects.
# Those objects are then indexed in their respective document and label index in the document store.
# The method can be used with any dataset in SQuAD format.
# We only use it to get the evaluation set labels and the corpus files.
document_store.add_eval_data(
    filename="data/tutorial5/nq_dev_subset_v2.json",
    doc_index=document_store.index,
    label_index=document_store.label_index,
    preprocessor=label_preprocessor,
)

# the evaluation set to evaluate the pipelines on
evaluation_set_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=True)

# Pipelines need files as input to be able to test different preprocessors.
# Even though this looks a bit cumbersome to write the documents back to files we gain a lot of evaluation potential and reproducibility.
docs = document_store.get_all_documents()
temp_dir = tempfile.TemporaryDirectory()
file_paths = []
for doc in docs:
    file_name = doc.id + ".txt"
    file_path = Path(temp_dir.name) / file_name
    file_paths.append(file_path)
    with open(file_path, "w") as f:
        f.write(doc.content)
file_metas = [d.meta for d in docs]

from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader, TextConverter
from haystack.pipelines import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
# helper function to create query and index pipeline
def create_pipelines(document_store, preprocessor, retriever, reader):
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, inputs=["Query"], name="Retriever")
    query_pipeline.add_node(component=reader, inputs=["Retriever"], name="Reader")
    index_pipeline = Pipeline()
    index_pipeline.add_node(component=TextConverter(), inputs=["File"], name="TextConverter")
    index_pipeline.add_node(component=preprocessor, inputs=["TextConverter"], name="Preprocessor")
    index_pipeline.add_node(component=retriever, inputs=["Preprocessor"], name="Retriever")
    index_pipeline.add_node(component=document_store, inputs=["Retriever"], name="DocumentStore")
    return query_pipeline, index_pipeline
# Name of the experiment in MLflow
EXPERIMENT_NAME = "haystack-tutorial-5"

document_store = ElasticsearchDocumentStore(host=host, index="sparse_index", recreate_index=True)
preprocessor = PreProcessor(
    split_length=200,
    split_overlap=0,
    split_respect_sentence_boundary=False,
    clean_empty_lines=False,
    clean_whitespace=False,
)
es_retriever = BM25Retriever(document_store=document_store)
reader = FARMReader("deepset/roberta-base-squad2", top_k=3, return_no_answer=True, batch_size=8)
query_pipeline, index_pipeline = create_pipelines(document_store, preprocessor, es_retriever, reader)

sparse_eval_result = Pipeline.execute_eval_run(
    index_pipeline=index_pipeline,
    query_pipeline=query_pipeline,
    evaluation_set_labels=evaluation_set_labels,
    corpus_file_paths=file_paths,
    corpus_file_metas=file_metas,
    experiment_name=EXPERIMENT_NAME,
    experiment_run_name="sparse",
    corpus_meta={"name": "nq_dev_subset_v2.json"},
    evaluation_set_meta={"name": "nq_dev_subset_v2.json"},
    pipeline_meta={"name": "sparse-pipeline"},
    add_isolated_node_eval=True,
    # experiment_tracking_tool="mlflow",                    # UNCOMMENT TO USE MLFLOW
    # experiment_tracking_uri="YOUR-MLFLOW-TRACKING-URI",   # UNCOMMENT TO USE MLFLOW
    reuse_index=True,
)

document_store = ElasticsearchDocumentStore(host=host, index="dense_index", recreate_index=True)
emb_retriever = EmbeddingRetriever(
    document_store=document_store,
    model_format="sentence_transformers",
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    batch_size=8,
)
query_pipeline, index_pipeline = create_pipelines(document_store, preprocessor, emb_retriever, reader)

dense_eval_result = Pipeline.execute_eval_run(
    index_pipeline=index_pipeline,
    query_pipeline=query_pipeline,
    evaluation_set_labels=evaluation_set_labels,
    corpus_file_paths=file_paths,
    corpus_file_metas=file_metas,
    experiment_name=EXPERIMENT_NAME,
    experiment_run_name="embedding",
    corpus_meta={"name": "nq_dev_subset_v2.json"},
    evaluation_set_meta={"name": "nq_dev_subset_v2.json"},
    pipeline_meta={"name": "embedding-pipeline"},
    add_isolated_node_eval=True,
    # experiment_tracking_tool="mlflow",                    # UNCOMMENT TO USE MLFLOW
    # experiment_tracking_uri="YOUR-MLFLOW-TRACKING-URI",   # UNCOMMENT TO USE MLFLOW
    reuse_index=True,
    answer_scope="context",
)
