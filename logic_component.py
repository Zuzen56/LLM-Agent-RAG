from pyscipopt import Model
from typing import List, Dict, Any
from haystack import Document


import os
from getpass import getpass
from haystack.utils import Secret
# from haystack.dataclasses import ChatMessage

HF_api_key = os.getenv("HF_API_KEY", None) or getpass("Enter HF API key:")

def validate_equation_solution(equation: str, answer: str) -> bool:
    """
    验证答案是否满足方程（示例：3x + 5 = 20 → x=5）
    """
    model = Model()
    x = model.addVar("x", vtype="C")
    try:
        # 将方程和答案转换为 PySCIPOpt 表达式
        equation_expr = eval(equation.replace("=", "=="))  # 示例：3*x + 5 == 20
        answer_expr = eval(answer)                          # 示例：x == 5
        model.addCons(equation_expr)
        model.addCons(~answer_expr)
        model.optimize()
        return model.getStatus() != "optimal"  # 若无解则答案正确
    except:
        return False


from haystack import component
from typing import Dict, Any

@component
class LogicValidator:
    @component.output_types(is_valid=bool, corrected_answer=str)
    def run(self, answer: str, documents: List[Document]):
        equation = documents[0].content if documents else ""
        is_valid = validate_equation_solution(equation, answer)
        if not is_valid:
            corrected_answer = solve_equation_with_pyscipopt(equation)
            return {"is_valid": False, "corrected_answer": corrected_answer}
        return {"is_valid": True, "corrected_answer": answer}


def solve_equation_with_pyscipopt(equation: str) -> str:
    model = Model()
    x = model.addVar("x", vtype="C")
    expr = eval(equation.replace("=", "=="))
    model.addCons(expr)
    model.optimize()
    if model.getStatus() == "optimal":
        return f"x = {model.getVal(x)}"
    return "无解"

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import HuggingFaceTGIChatGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

template = [
    ChatMessage.from_system(
        """
Answer the questions based on the given context.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}
Question: {{ question }}
Answer:
"""
    )
]

document_store = InMemoryDocumentStore()


pipeline = Pipeline()
pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
pipeline.add_component("generator", HuggingFaceTGIChatGenerator(model="Qwen/Qwen2.5-72B-Instruct",token=Secret.from_token(HF_api_key)))
pipeline.add_component("logic_validator", LogicValidator())

# 连接节点
pipeline.connect("embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "generator.messages")


@component
class ExtractText:
    @component.output_types(output=str)
    def run(self, replies: List[ChatMessage]):
        return {"output": replies[0].content if replies else ""}

pipeline.add_component("extract_text", ExtractText())
pipeline.connect("generator.replies", "extract_text.replies")
pipeline.connect("extract_text.output", "logic_validator.answer")
pipeline.connect("retriever.documents", "logic_validator.documents")

# 运行管道
question = "解方程 3x + 5 = 20，求 x。"
results = pipeline.run({
    "embedder": {"text": question},
    "prompt_builder": {"query": question},
    "logic_validator": {"context": {"equation": "3x +5=20"}}  # 从问题中提取方程
})

# 输出结果
if results["logic_validator"]["is_valid"]:
    print("正确答案:", results["logic_validator"]["corrected_answer"])
else:
    print("原始答案有误，已修正为:", results["logic_validator"]["corrected_answer"])