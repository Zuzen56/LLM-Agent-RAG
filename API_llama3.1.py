from openai import OpenAI
#llama的api
client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': '你好，你是谁？',
        }
    ],
    model='llama3.1:latest',
)

print(chat_completion.choices[0].message.content)