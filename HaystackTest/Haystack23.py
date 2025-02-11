from haystack.document_stores import InMemoryDocumentStore
from datasets import load_dataset

remote_dataset = load_dataset("Tuana/presidents", split="train")

document_store = InMemoryDocumentStore(use_bm25=True)
document_store.write_documents(remote_dataset)

from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1", use_gpu=True
)
document_store.update_embeddings(retriever=retriever)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
presidents_qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)

from haystack.utils import print_answers

# result = presidents_qa.run("Who was the 1st president of the USA?")
result = presidents_qa.run("What year was the 1st president of the USA born?")

print_answers(result, "minimum")

import os
from getpass import getpass

api_key = os.getenv("OPENAI_API_KEY", None) or getpass("Enter OpenAI API key:")

from haystack.agents import Agent
from haystack.nodes import PromptNode

prompt_node = PromptNode(model_name_or_path="gpt-3.5-turbo-instruct", api_key=api_key, stop_words=["Observation:"])
agent = Agent(prompt_node=prompt_node)


from haystack.agents import Tool

search_tool = Tool(
    name="USA_Presidents_QA",
    pipeline_or_node=presidents_qa,
    description="useful for when you need to answer questions related to the presidents of the USA.",
    output_variable="answers",
)
agent.add_tool(search_tool)

result = agent.run("What year was the 1st president of the USA born?")

print(result["transcript"].split("---")[0])