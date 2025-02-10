from haystack.document_stores import FAISSDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

import sqlalchemy
print(sqlalchemy.__version__)  # 输出当前版本
document_store = FAISSDocumentStore(sql_url="sqlite:///", faiss_index_factory_str="Flat")


# Let's first get some files that we want to use
doc_dir = "data/tutorial6"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt6.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Convert files to dicts
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(docs)



retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
# Important:
# Now that we initialized the Retriever, we need to call update_embeddings() to iterate over all
# previously indexed documents and update their embedding representation.
# While this can be a time consuming operation (depending on the corpus size), it only needs to be done once.
# At query time, we only need to embed the query and compare it to the existing document embeddings, which is very fast.
document_store.update_embeddings(retriever)



reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)



pipe = ExtractiveQAPipeline(reader, retriever)

prediction = pipe.run(
    query="Who created the Dothraki vocabulary?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)

from haystack.utils import print_answers


print_answers(prediction, details="minimum")
