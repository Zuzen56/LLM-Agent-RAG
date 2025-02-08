import os
from haystack import Pipeline, PredefinedPipeline
from openai import OpenAI

# 设置本地模型的API密钥和URL
os.environ["OPENAI_API_KEY"] = "ollama"
base_url = 'http://localhost:11434/v1/'

# 使用本地文件路径
local_file_path = "davinci.txt"

# #测试文件是否能正常读取
# with open(local_file_path,"r",encoding="utf-8") as f:
#     text = f.read()
#     print(text)

# 创建索引管道
indexing_pipeline = Pipeline.from_template(PredefinedPipeline.INDEXING)
indexing_pipeline.run(data={"sources": [local_file_path]})

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