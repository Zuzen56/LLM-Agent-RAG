from haystack.utils import fetch_archive_from_http


# This fetches some sample files to work with
doc_dir = "data/tutorial16"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/preprocessing_tutorial16.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

from haystack.nodes import PreProcessor
from haystack.utils import convert_files_to_docs

# note that you can also use the document classifier before applying the PreProcessor, e.g. before splitting your documents
all_docs = convert_files_to_docs(dir_path=doc_dir)
preprocessor_sliding_window = PreProcessor(split_overlap=3, split_length=10, split_respect_sentence_boundary=False)
docs_sliding_window = preprocessor_sliding_window.process(all_docs)

from haystack.nodes import TransformersDocumentClassifier


doc_classifier = TransformersDocumentClassifier(
    model_name_or_path="cross-encoder/nli-distilroberta-base",
    task="zero-shot-classification",
    labels=["music", "natural language processing", "history"],
    batch_size=16,
)

# classify using gpu, batch_size makes sure we do not run out of memory
classified_docs = doc_classifier.predict(docs_sliding_window)

# let's see how it looks: there should be a classification result in the meta entry containing labels and scores.
print(classified_docs[0].to_dict())

import os
import time

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore

# Wait 30 seconds only to be sure Elasticsearch is ready before continuing
time.sleep(30)

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(host=host, username="", password="", index="document")
# Now, let's write the docs to our DB.
document_store.delete_all_documents()
document_store.write_documents(classified_docs)
# check if indexed docs contain classification results
test_doc = document_store.get_all_documents()[0]
print(
    f'document {test_doc.id} with content \n\n{test_doc.content}\n\nhas label {test_doc.meta["classification"]["label"]}'
)

# Initialize QA-Pipeline
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader, BM25Retriever


retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)
## Voilà! Ask a question while filtering for "music"-only documents
prediction = pipe.run(
    query="What is heavy metal?",
    params={"Retriever": {"top_k": 10, "filters": {"classification.label": ["music"]}}, "Reader": {"top_k": 5}},
)
from haystack.utils import print_answers


print_answers(prediction, details="high")

from pathlib import Path
from haystack.pipelines import Pipeline
from haystack.nodes import TextConverter, PreProcessor, FileTypeClassifier, PDFToTextConverter, DocxToTextConverter


file_type_classifier = FileTypeClassifier()
text_converter = TextConverter()
pdf_converter = PDFToTextConverter()
docx_converter = DocxToTextConverter()

indexing_pipeline_with_classification = Pipeline()
indexing_pipeline_with_classification.add_node(
    component=file_type_classifier, name="FileTypeClassifier", inputs=["File"]
)
indexing_pipeline_with_classification.add_node(
    component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"]
)
indexing_pipeline_with_classification.add_node(
    component=pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"]
)
indexing_pipeline_with_classification.add_node(
    component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"]
)
indexing_pipeline_with_classification.add_node(
    component=preprocessor_sliding_window,
    name="Preprocessor",
    inputs=["TextConverter", "PdfConverter", "DocxConverter"],
)
indexing_pipeline_with_classification.add_node(
    component=doc_classifier, name="DocumentClassifier", inputs=["Preprocessor"]
)
indexing_pipeline_with_classification.add_node(
    component=document_store, name="DocumentStore", inputs=["DocumentClassifier"]
)
# Uncomment the following to generate the pipeline image
# indexing_pipeline_with_classification.draw("index_time_document_classifier.png")

document_store.delete_documents()
txt_files = [f for f in Path(doc_dir).iterdir() if f.suffix == ".txt"]
pdf_files = [f for f in Path(doc_dir).iterdir() if f.suffix == ".pdf"]
docx_files = [f for f in Path(doc_dir).iterdir() if f.suffix == ".docx"]
indexing_pipeline_with_classification.run(file_paths=txt_files)
indexing_pipeline_with_classification.run(file_paths=pdf_files)
indexing_pipeline_with_classification.run(file_paths=docx_files)

document_store.get_all_documents()[0]
# we can store this pipeline and use it from the REST-API
indexing_pipeline_with_classification.save_to_yaml("indexing_pipeline_with_classification.yaml")




