from typing import List
from pydantic import BaseModel


class City(BaseModel):
    name: str
    country: str
    population: int


class CitiesData(BaseModel):
    cities: List[City]

json_schema = CitiesData.schema_json(indent=2)

import json
import random
import pydantic
from pydantic import ValidationError
from typing import Optional, List
from colorama import Fore
from haystack import component
from haystack.dataclasses import ChatMessage


# Define the component input parameters
@component
class OutputValidator:
    def __init__(self, pydantic_model: pydantic.BaseModel):
        self.pydantic_model = pydantic_model
        self.iteration_counter = 0

    # Define the component output
    @component.output_types(valid_replies=List[str], invalid_replies=Optional[List[str]], error_message=Optional[str])
    def run(self, replies: List[ChatMessage]):

        self.iteration_counter += 1

        ## Try to parse the LLM's reply ##
        # If the LLM's reply is a valid object, return `"valid_replies"`
        try:
            # output_dict = json.loads(replies[0].content)

            start_index = replies[0].content.find('{')
            json_str = replies[0].content[start_index:]
            output_dict = json.loads(json_str)
            print(output_dict)


            self.pydantic_model.parse_obj(output_dict)
            print(
                Fore.GREEN
                + f"OutputValidator at Iteration {self.iteration_counter}: Valid JSON from LLM - No need for looping: {replies[0]}"
            )
            return {"valid_replies": replies}

        # If the LLM's reply is corrupted or not valid, return "invalid_replies" and the "error_message" for LLM to try again
        except (ValueError, ValidationError) as e:
            print(
                Fore.RED
                + f"OutputValidator at Iteration {self.iteration_counter}: Invalid JSON from LLM - Let's try again.\n"
                f"Output from LLM:\n {replies[0]} \n"
                f"Error from OutputValidator: {e}"
            )
            return {"invalid_replies": replies, "error_message": str(e)}

output_validator = OutputValidator(pydantic_model=CitiesData)

from haystack.components.builders import ChatPromptBuilder


prompt_template = [
    ChatMessage.from_user(
        """
Create a JSON object from the information present in this passage: {{passage}}.
Only use information that is present in the passage. Follow this JSON schema, but only return the actual instances without any additional schema definition:
{{schema}}
Make sure your response is a dict and not a list.
{% if invalid_replies and error_message %}
  You already created the following output in a previous attempt: {{invalid_replies}}
  However, this doesn't comply with the format requirements from above and triggered this Python exception: {{error_message}}
  Correct the output and try again. Just return the corrected output without any extra explanations.
{% endif %}
"""
    )
]
prompt_builder = ChatPromptBuilder(template=prompt_template)

# import os
# from getpass import getpass
#
# from haystack.components.generators.chat import OpenAIChatGenerator
#
# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
# chat_generator = OpenAIChatGenerator()

import os
from getpass import getpass
from haystack.utils import Secret
from haystack.components.generators.chat import HuggingFaceTGIChatGenerator

api_key = os.getenv("HF_API_KEY", None) or getpass("Enter HF API key:")

chat_generator = HuggingFaceTGIChatGenerator(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",token=Secret.from_token(api_key))


from haystack import Pipeline
print("start")
pipeline = Pipeline(max_loops_allowed=100)
print("suceess")
# Add components to your pipeline
pipeline.add_component(instance=prompt_builder, name="prompt_builder")
pipeline.add_component(instance=chat_generator, name="llm")
pipeline.add_component(instance=output_validator, name="output_validator")

# Now, connect the components to each other
pipeline.connect("prompt_builder.prompt", "llm.messages")
pipeline.connect("llm.replies", "output_validator")
# If a component has more than one output or input, explicitly specify the connections:
pipeline.connect("output_validator.invalid_replies", "prompt_builder.invalid_replies")
pipeline.connect("output_validator.error_message", "prompt_builder.error_message")

print("start")
pipeline.draw("auto-correct-pipeline.png")
print("suceess")
passage = "Berlin is the capital of Germany. It has a population of 3,850,809. Paris, France's capital, has 2.161 million residents. Lisbon is the capital and the largest city of Portugal with the population of 504,718."
result = pipeline.run({"prompt_builder": {"passage": passage, "schema": json_schema}})

# valid_reply = result["output_validator"]["valid_replies"][0].content
# valid_json = json.loads(valid_reply)
# print(valid_json)

start_index = result["output_validator"]["valid_replies"][0].content.find('{')
json_str = result["output_validator"]["valid_replies"][0].content[start_index:]
valid_json = json.loads(json_str)
print(valid_json)

