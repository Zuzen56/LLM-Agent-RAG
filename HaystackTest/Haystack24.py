import os
from getpass import getpass

model_api_key = os.getenv("HF_API_KEY", None) or getpass("Enter HF API key:")

from haystack.nodes import PromptNode

prompt_node = PromptNode(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta", api_key=model_api_key, max_length=256, stop_words=["Human"]
)

from haystack.agents.memory import ConversationSummaryMemory

summary_memory = ConversationSummaryMemory(prompt_node)

from haystack.agents.conversational import ConversationalAgent

conversational_agent = ConversationalAgent(prompt_node=prompt_node, memory=summary_memory)

conversational_agent.run("Tell me three most interesting things about Istanbul, Turkey")
conversational_agent.run("Can you elaborate on the second item?")
conversational_agent.run("Can you turn this info into a twitter thread?")
