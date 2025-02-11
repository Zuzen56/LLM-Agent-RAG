from datasets import load_dataset
from haystack.document_stores import InMemoryDocumentStore

dataset = load_dataset("anakin87/medrag-pubmed-chunk", split="train")

document_store = InMemoryDocumentStore(use_bm25=True, embedding_dim=384)

from haystack.schema import Document

docs = []
for doc in dataset:
    docs.append(
        Document(content=doc["contents"], meta={"title": doc["title"], "abstract": doc["content"], "pmid": doc["id"]})
    )

from haystack.nodes import PreProcessor

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=512,
    split_overlap=32,
    split_respect_sentence_boundary=True,
)
docs_to_index = preprocessor.process(docs)

from haystack.nodes import EmbeddingRetriever, BM25Retriever

sparse_retriever = BM25Retriever(document_store=document_store)
dense_retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=True,
    scale_score=False,
)

document_store.delete_documents()
document_store.write_documents(docs_to_index)
document_store.update_embeddings(retriever=dense_retriever)

from haystack.nodes import JoinDocuments, SentenceTransformersRanker

join_documents = JoinDocuments(join_mode="concatenate")
rerank = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")

from haystack.pipelines import Pipeline

pipeline = Pipeline()
pipeline.add_node(component=sparse_retriever, name="SparseRetriever", inputs=["Query"])
pipeline.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
pipeline.add_node(component=join_documents, name="JoinDocuments", inputs=["SparseRetriever", "DenseRetriever"])
pipeline.add_node(component=rerank, name="ReRanker", inputs=["JoinDocuments"])

prediction = pipeline.run(
    query="apnea in infants",
    params={
        "SparseRetriever": {"top_k": 10},
        "DenseRetriever": {"top_k": 10},
        "JoinDocuments": {"top_k_join": 15},  # comment for debug
        # "JoinDocuments": {"top_k_join": 15, "debug":True}, #uncomment for debug
        "ReRanker": {"top_k": 5},
    },
)

def pretty_print_results(prediction):
    for doc in prediction["documents"]:
        print(doc.meta["title"], "\t", doc.score)
        print(doc.meta["abstract"])
        print("\n", "\n")
pretty_print_results(prediction)






