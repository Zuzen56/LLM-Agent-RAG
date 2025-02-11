
from datasets import load_dataset
from haystack.document_stores import InMemoryDocumentStore

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")

document_store = InMemoryDocumentStore(use_bm25=True)
document_store.write_documents(dataset)

import os
from getpass import getpass
model_api_key = os.getenv("HF_API_KEY", None) or getpass("Enter HF API key:")



from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline

retriever = BM25Retriever(document_store=document_store, top_k=3)

prompt_template = PromptTemplate(
    prompt="""
    Answer the question truthfully based solely on the given documents. If the documents do not contain the answer to the question, say that answering is not possible given the available information. Your answer should be no longer than 50 words.
    Documents:{join(documents)}
    Question:{query}
    Answer:
    """,
    output_parser=AnswerParser(),
)

prompt_node = PromptNode(
    model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", api_key=model_api_key, default_prompt_template=prompt_template
)

generative_pipeline = Pipeline()
generative_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
generative_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

from haystack.utils import print_answers

response = generative_pipeline.run("What does Rhodes Statue look like?")
print_answers(response, details="minimum")


response = generative_pipeline.run("What does Taylor Swift look like?")
print_answers(response, details="minimum")


from haystack.agents import Tool

search_tool = Tool(
    name="seven_wonders_search",
    pipeline_or_node=generative_pipeline,
    description="useful for when you need to answer questions about the seven wonders of the world",
    output_variable="answers",
)

from haystack.nodes import PromptNode

agent_prompt_node = PromptNode(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    api_key=model_api_key,
    max_length=256,
    stop_words=["Observation:"],
    model_kwargs={"temperature": 0.5},
)



from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode

memory_prompt_node = PromptNode(
    "philschmid/bart-large-cnn-samsum", max_length=256, model_kwargs={"task_name": "text2text-generation"}
)
memory = ConversationSummaryMemory(memory_prompt_node, prompt_template="{chat_transcript}")


agent_prompt = """
In the following conversation, a human user interacts with an AI Agent. The human user poses questions, and the AI Agent goes through several steps to provide well-informed answers.
The AI Agent must use the available tools to find the up-to-date information. The final answer to the question should be truthfully based solely on the output of the tools. The AI Agent should ignore its knowledge when answering the questions.
The AI Agent has access to these tools:
{tool_names_with_descriptions}

The following is the previous conversation between a human and The AI Agent:
{memory}

AI Agent responses must start with one of the following:

Thought: [the AI Agent's reasoning process]
Tool: [tool names] (on a new line) Tool Input: [input as a question for the selected tool WITHOUT quotation marks and on a new line] (These must always be provided together and on separate lines.)
Observation: [tool's result]
Final Answer: [final answer to the human user's question]
When selecting a tool, the AI Agent must provide both the "Tool:" and "Tool Input:" pair in the same response, but on separate lines.

The AI Agent should not ask the human user for additional information, clarification, or context.
If the AI Agent cannot find a specific answer after exhausting available tools and approaches, it answers with Final Answer: inconclusive

Question: {query}
Thought:
{transcript}
"""



from haystack.agents import AgentStep, Agent


def resolver_function(query, agent, agent_step):
    return {
        "query": query,
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
        "memory": agent.memory.load(),
    }


from haystack.agents.base import Agent, ToolsManager

conversational_agent = Agent(
    agent_prompt_node,
    prompt_template=agent_prompt,
    prompt_parameters_resolver=resolver_function,
    memory=memory,
    tools_manager=ToolsManager([search_tool]),
)

conversational_agent.run("What did Rhodes Statue look like?")

conversational_agent.run("When did it collapse?")

conversational_agent.run("How tall was it?")

conversational_agent.run("How long did it stand?")




