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

from haystack.components.embedders import SentenceTransformersTextEmbedder

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

retriever = InMemoryEmbeddingRetriever(document_store)

from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

template = [
    ChatMessage.from_user(
        """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]

prompt_builder = ChatPromptBuilder(template=template)

import os
from getpass import getpass
from haystack.utils import Secret
from haystack.components.generators.chat import HuggingFaceTGIChatGenerator

api_key = os.getenv("HF_API_KEY", None) or getpass("Enter HF API key:")

chat_generator = HuggingFaceTGIChatGenerator(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",token=Secret.from_token(api_key))



from haystack import Pipeline

basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)
# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

question = "What does Rhodes Statue look like?"

response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})


print(response["llm"]["replies"][0].content)
