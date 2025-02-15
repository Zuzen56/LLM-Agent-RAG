import os
import dashscope
from dashscope.api_entities.dashscope_response import Message

class ModelProvider(object):
    def __init__(self):
        self.api_key = os.environ.get("API_KEY")
        self.model_name = os.environ.get("MODEL_NAME")
        self.client = dashscope.Generation()
        self.max_retry_time = 3

    def chat(self, prompt,chat_history):
        cur_retry_time = 0
        messages = []
        user_prompt = prompt
        while cur_retry_time < self.max_retry_time:
            cur_retry_time += 1
            try:
                messages = [Message(role="system", content=prompt)]
                for his in chat_history:
                    messages.append(Message(role="user", content=his[0]))
                    messages.append(Message(role="assistant", content=his[1]))
                messages.append(Message(role="user", content=user_prompt))
                response = self._client_call(
                    model=self.model_name,
                    api_key=self.api_key,
                    messages=messages
                )
                """
                {
                    
                }
                """
            except Exception as err:
                print("调用大模型出错：{}".format(err))


