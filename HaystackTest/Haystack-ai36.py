#此文件在result内容输出有问题
#猜测可能是qwen模型调用时多了一个参数，或者promt文本内容格式有问题

import os
from getpass import getpass
from haystack.components.builders import AnswerBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever


from haystack.utils import Secret


HF_api_key = os.getenv("HF_API_KEY", None) or getpass("Enter HF API key:")
Serper_api_key = os.getenv("SERPERDEV_API_KEY", None) or getpass("Enter Serper API key:")



from haystack.dataclasses import Document

documents = [
    Document(
        content="""Munich, the vibrant capital of Bavaria in southern Germany, exudes a perfect blend of rich cultural
                                heritage and modern urban sophistication. Nestled along the banks of the Isar River, Munich is renowned
                                for its splendid architecture, including the iconic Neues Rathaus (New Town Hall) at Marienplatz and
                                the grandeur of Nymphenburg Palace. The city is a haven for art enthusiasts, with world-class museums like the
                                Alte Pinakothek housing masterpieces by renowned artists. Munich is also famous for its lively beer gardens, where
                                locals and tourists gather to enjoy the city's famed beers and traditional Bavarian cuisine. The city's annual
                                Oktoberfest celebration, the world's largest beer festival, attracts millions of visitors from around the globe.
                                Beyond its cultural and culinary delights, Munich offers picturesque parks like the English Garden, providing a
                                serene escape within the heart of the bustling metropolis. Visitors are charmed by Munich's warm hospitality,
                                making it a must-visit destination for travelers seeking a taste of both old-world charm and contemporary allure."""
    )
]


from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import HuggingFaceTGIChatGenerator

prompt_template = [
    ChatMessage.from_user(
        """
Answer the following query given the documents.
If the answer is not contained within the documents reply with 'no_answer'
Query: {{query}}
Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}
"""
    )
]

prompt_builder = ChatPromptBuilder(template=prompt_template)
llm = HuggingFaceTGIChatGenerator(model="Qwen/Qwen2.5-72B-Instruct",token=Secret.from_token(HF_api_key))

from haystack.components.websearch.serper_dev import SerperDevWebSearch

prompt_for_websearch = [
    ChatMessage.from_user(
        """
Answer the following query given the documents retrieved from the web.
Your answer shoud indicate that your answer was generated from websearch.

Query: {{query}}
Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}
"""
    )
]

websearch = SerperDevWebSearch(api_key=Secret.from_token(Serper_api_key))

prompt_builder_for_websearch = ChatPromptBuilder(template=prompt_for_websearch)
llm_for_websearch = HuggingFaceTGIChatGenerator(model="Qwen/Qwen2.5-72B-Instruct",token=Secret.from_token(HF_api_key))

from haystack.components.routers import ConditionalRouter

routes = [
    {
        "condition": "{{'no_answer' in replies[0].text}}",
        "output": "{{query}}",
        "output_name": "go_to_websearch",
        "output_type": str,
    },
    {
        "condition": "{{'no_answer' not in replies[0].text}}",
        "output": "{{replies[0].text}}",
        "output_name": "answer",
        "output_type": str,
    },
]

router = ConditionalRouter(routes)

from haystack import Pipeline

pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.add_component("router", router)
pipe.add_component("websearch", websearch)
pipe.add_component("prompt_builder_for_websearch", prompt_builder_for_websearch)
pipe.add_component("llm_for_websearch", llm_for_websearch)

pipe.connect("prompt_builder.prompt", "llm.messages")
pipe.connect("llm.replies", "router.replies")
pipe.connect("router.go_to_websearch", "websearch.query")
pipe.connect("router.go_to_websearch", "prompt_builder_for_websearch.query")
pipe.connect("websearch.documents", "prompt_builder_for_websearch.documents")
pipe.connect("prompt_builder_for_websearch", "llm_for_websearch")

pipe.draw("pipe.png")

query = "Where is Munich?"

result = pipe.run({"prompt_builder": {"query": query, "documents": documents}, "router": {"query": query}})

# Print the `answer` coming from the ConditionalRouter
print(result["router"]["answer"])



query = "How many people live in Munich?"

result = pipe.run({"prompt_builder": {"query": query, "documents": documents}, "router": {"query": query}})

# Print the `replies` generated using the web searched Documents
print(result["llm_for_websearch"]["replies"])

result