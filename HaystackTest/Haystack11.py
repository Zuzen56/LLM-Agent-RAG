from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore(use_bm25=True)

from haystack.utils import fetch_archive_from_http, convert_files_to_docs, clean_wiki_text

# Download and prepare data - 517 Wikipedia articles for Game of Thrones
doc_dir = "data/tutorial11"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt11.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# convert files to dicts containing documents that can be indexed to our datastore
got_docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.delete_documents()
document_store.write_documents(got_docs)

from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader

bm25_retriever = BM25Retriever()

from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

p_retrieval = DocumentSearchPipeline(bm25_retriever)

from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader

# Initialize Sparse Retriever
bm25_retriever = BM25Retriever(document_store=document_store)

# Initialize embedding Retriever
embedding_retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

# Initialize Reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers

p_extractive_premade = ExtractiveQAPipeline(reader=reader, retriever=bm25_retriever)
res = p_extractive_premade.run(
    query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)
print_answers(res, details="minimum")

from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

p_retrieval = DocumentSearchPipeline(embedding_retriever)
res = p_retrieval.run(query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}})
print_documents(res, max_text_len=200)

from haystack.pipelines import Pipeline

# Custom built extractive QA pipeline
p_extractive = Pipeline()
p_extractive.add_node(component=bm25_retriever, name="Retriever", inputs=["Query"])
p_extractive.add_node(component=reader, name="Reader", inputs=["Retriever"])

# Now we can run it
res = p_extractive.run(query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}})
print_answers(res, details="minimum")


from haystack.nodes import JoinDocuments

# Create ensembled pipeline
p_ensemble = Pipeline()
p_ensemble.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
p_ensemble.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
p_ensemble.add_node(
    component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["BM25Retriever", "EmbeddingRetriever"]
)
p_ensemble.add_node(component=reader, name="Reader", inputs=["JoinResults"])

# Uncomment the following to generate the pipeline image
# p_ensemble.draw("pipeline_ensemble.png")

# Run pipeline
res = p_ensemble.run(
    query="Who is the father of Arya Stark?", params={"EmbeddingRetriever": {"top_k": 5}, "BM25Retriever": {"top_k": 5}}
)
print_answers(res, details="minimum")

from haystack import BaseComponent
from typing import Optional, List


class CustomNode(BaseComponent):
    outgoing_edges = 1

    def run(self, query: str, my_optional_param: Optional[int]):
        # process the inputs
        output = {"my_output": ...}
        return output, "output_1"

    def run_batch(self, queries: List[str], my_optional_param: Optional[int]):
        # process the inputs
        output = {"my_output": ...}
        return output, "output_1"

class CustomQueryClassifier(BaseComponent):
    outgoing_edges = 2

    def run(self, query: str):
        if "?" in query:
            return {}, "output_2"
        else:
            return {}, "output_1"

    def run_batch(self, queries: List[str]):
        split = {"output_1": {"queries": []}, "output_2": {"queries": []}}
        for query in queries:
            if "?" in query:
                split["output_2"]["queries"].append(query)
            else:
                split["output_1"]["queries"].append(query)

        return split, "split"


# Here we build the pipeline
p_classifier = Pipeline()
p_classifier.add_node(component=CustomQueryClassifier(), name="QueryClassifier", inputs=["Query"])
p_classifier.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["QueryClassifier.output_1"])
p_classifier.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["QueryClassifier.output_2"])
p_classifier.add_node(component=reader, name="QAReader", inputs=["BM25Retriever", "EmbeddingRetriever"])
# Uncomment the following to generate the pipeline image
# p_classifier.draw("pipeline_classifier.png")

# Run only the dense retriever on the full sentence query
res_1 = p_classifier.run(query="Who is the father of Arya Stark?")
print("Embedding Retriever Results" + "\n" + "=" * 15)
print_answers(res_1)

# Run only the sparse retriever on a keyword based query
res_2 = p_classifier.run(query="Arya Stark father")
print("BM25Retriever Results" + "\n" + "=" * 15)
print_answers(res_2)

# 1) You can set the `debug` attribute of a given node.
bm25_retriever.debug = True

# 2) You can provide `debug` as a parameter when running your pipeline
result = p_classifier.run(query="Who is the father of Arya Stark?", params={"BM25Retriever": {"debug": True}})

# 3) You can provide the `debug` paramter to all nodes in your pipeline
result = p_classifier.run(query="Who is the father of Arya Stark?", params={"debug": True})

result["_debug"]
