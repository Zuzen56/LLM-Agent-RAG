#此文件在131行左右有问题，需要修改
#问题原因是llm_evaluator.py文件中的代码默认调用openai的api，需要修改为调用huggingface的api
#以后用到了再改吧，或者想办法搞个openai 的token

from datasets import load_dataset
from haystack import Document

dataset = load_dataset("vblagoje/PubMedQA_instruction", split="train")
dataset = dataset.select(range(1000))
all_documents = [Document(content=doc["context"]) for doc in dataset]
all_questions = [doc["instruction"] for doc in dataset]
all_ground_truth_answers = [doc["response"] for doc in dataset]

from typing import List
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

document_store = InMemoryDocumentStore()

document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

indexing = Pipeline()
indexing.add_component(instance=document_embedder, name="document_embedder")
indexing.add_component(instance=document_writer, name="document_writer")

indexing.connect("document_embedder.documents", "document_writer.documents")

indexing.run({"document_embedder": {"documents": all_documents}})


import os
from getpass import getpass
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever


from haystack.utils import Secret
from haystack.components.generators.chat import HuggingFaceTGIChatGenerator

api_key = os.getenv("HF_API_KEY", None) or getpass("Enter HF API key:")

chat_generator = HuggingFaceTGIChatGenerator(model="Qwen/Qwen2.5-72B-Instruct",token=Secret.from_token(api_key))

template = [
    ChatMessage.from_user(
        """
        You have to answer the following question based on the given context information only.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """
    )
]

from haystack import component
@component
class ChatMessageToString:
    @component.output_types(replies=List[str])
    def run(self, messages: List[ChatMessage]):
        return {"replies": [msg.content for msg in messages]}

rag_pipeline = Pipeline()
rag_pipeline.add_component(
    "query_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
rag_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=3))
rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
rag_pipeline.add_component("generator", chat_generator)
rag_pipeline.add_component("message_converter", ChatMessageToString())
rag_pipeline.add_component("answer_builder", AnswerBuilder())

rag_pipeline.connect("query_embedder", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "generator.messages")
rag_pipeline.connect("generator.replies", "message_converter.messages")  # 先转换类型
rag_pipeline.connect("message_converter.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")

question = "Do high levels of procalcitonin in the early phase after pediatric liver transplantation indicate poor postoperative outcome?"

response = rag_pipeline.run(
    {
        "query_embedder": {"text": question},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
    }
)
print(response["answer_builder"]["answers"][0].data)

import random

questions, ground_truth_answers, ground_truth_docs = zip(
    *random.sample(list(zip(all_questions, all_ground_truth_answers, all_documents)), 25)
)

rag_answers = []
retrieved_docs = []

for question in list(questions):
    response = rag_pipeline.run(
        {
            "query_embedder": {"text": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }
    )
    print(f"Question: {question}")
    print("Answer from pipeline:")
    print(response["answer_builder"]["answers"][0].data)
    print("\n-----------------------------------\n")

    rag_answers.append(response["answer_builder"]["answers"][0].data)
    retrieved_docs.append(response["answer_builder"]["answers"][0].documents)

from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
from haystack.components.evaluators.faithfulness import FaithfulnessEvaluator
from haystack.components.evaluators.sas_evaluator import SASEvaluator

eval_pipeline = Pipeline()
eval_pipeline.add_component("doc_mrr_evaluator", DocumentMRREvaluator())
eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
eval_pipeline.add_component("sas_evaluator", SASEvaluator(model="sentence-transformers/all-MiniLM-L6-v2"))

results = eval_pipeline.run(
    {
        "doc_mrr_evaluator": {
            "ground_truth_documents": list([d] for d in ground_truth_docs),
            "retrieved_documents": retrieved_docs,
        },
        "faithfulness": {
            "questions": list(questions),
            "contexts": list([d.content] for d in ground_truth_docs),
            "predicted_answers": rag_answers,
        },
        "sas_evaluator": {"predicted_answers": rag_answers, "ground_truth_answers": list(ground_truth_answers)},
    }
)

from haystack.evaluation.eval_run_result import EvaluationRunResult

inputs = {
    "question": list(questions),
    "contexts": list([d.content] for d in ground_truth_docs),
    "answer": list(ground_truth_answers),
    "predicted_answer": rag_answers,
}

evaluation_result = EvaluationRunResult(run_name="pubmed_rag_pipeline", inputs=inputs, results=results)
evaluation_result.score_report()

results_df = evaluation_result.to_pandas()
results_df

import pandas as pd

top_3 = results_df.nlargest(3, "sas_evaluator")
bottom_3 = results_df.nsmallest(3, "sas_evaluator")
pd.concat([top_3, bottom_3])


