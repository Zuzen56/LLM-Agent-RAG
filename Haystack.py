import os
from haystack import Pipeline, PredefinedPipeline
from openai import OpenAI

# 设置本地模型的API密钥和URL
os.environ["OPENAI_API_KEY"] = "ollama"
base_url = 'http://localhost:11434/v1/'

# 下载文件
import urllib.request
urllib.request.urlretrieve("https://archive.org/stream/leonardodavinci00brocrich/leonardodavinci00brocrich_djvu.txt", "davinci.txt")

# 创建索引管道
indexing_pipeline = Pipeline.from_template(PredefinedPipeline.INDEXING)
indexing_pipeline.run(data={"sources": ["davinci.txt"]})

# 创建RAG管道
rag_pipeline = Pipeline.from_template(PredefinedPipeline.RAG)

# 查询
query = "How old was he when he died?"
result = rag_pipeline.run(data={"prompt_builder": {"query": query}, "text_embedder": {"text": query}})

# 调用本地模型
client = OpenAI(base_url=base_url, api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="deepseek-r1:14b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": query},
    ]
)

print(response.choices[0].message.content)