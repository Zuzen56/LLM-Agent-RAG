from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
import pandas as pd
from haystack.utils import fetch_archive_from_http
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers


document_store = InMemoryDocumentStore()

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=True,
    scale_score=False,
)


# Download
doc_dir = "data/tutorial4"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/small_faq_covid.csv.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Get dataframe with columns "question", "answer" and some custom metadata
df = pd.read_csv(f"{doc_dir}/small_faq_covid.csv")
# Minimal cleaning
df.fillna(value="", inplace=True)
df["question"] = df["question"].apply(lambda x: x.strip())
print(df.head())

# Create embeddings for our questions from the FAQs
# In contrast to most other search use cases, we don't create the embeddings here from the content of our documents,
# but rather from the additional text field "question" as we want to match "incoming question" <-> "stored question".
questions = list(df["question"].values)
df["embedding"] = retriever.embed_queries(queries=questions).tolist()
df = df.rename(columns={"question": "content"})

# Convert Dataframe to list of dicts and index them in our DocumentStore
docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)




pipe = FAQPipeline(retriever=retriever)



# Run any question and change top_k to see more or less answers
prediction = pipe.run(query="How is the virus spreading?", params={"Retriever": {"top_k": 1}})

print_answers(prediction, details="medium")