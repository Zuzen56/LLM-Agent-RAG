from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore(use_bm25=True)



from datasets import load_dataset

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")

document_store.write_documents(dataset)

from haystack.nodes import BM25Retriever

retriever = BM25Retriever(document_store=document_store, top_k=2)

# import os
# from getpass import getpass
# openai_api_key = os.getenv("OPENAI_API_KEY", None) or getpass("Enter OpenAI API key:")

import os
from getpass import getpass
model_api_key = os.getenv("HF_API_KEY", None) or getpass("Enter HF API key:")


from haystack.nodes import PromptNode, PromptTemplate, AnswerParser

rag_prompt = PromptTemplate(
    prompt="""Synthesize a comprehensive answer from the following text for the given question.
                             Provide a clear and concise response that summarizes the key points and information presented in the text.
                             Your answer should be in your own words and be no longer than 50 words.
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
    output_parser=AnswerParser(),
)

# prompt_node = PromptNode(model_name_or_path="gpt-3.5-turbo", api_key=openai_api_key, default_prompt_template=rag_prompt)

prompt_node = PromptNode(
    model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", api_key=model_api_key, default_prompt_template=rag_prompt
)


from haystack.pipelines import Pipeline

pipe = Pipeline()
pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

output = pipe.run(query="What does Rhodes Statue look like?")

print(output["answers"][0].answer)