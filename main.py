import time
from tools import tools_map
from promt import gen_prompt,user_promt
from model_provider import ModelProvider
from dotenv import load_dotenv

load_dotenv()



mp = ModelProvider()
def parser_thoughts(response):
    """
            response: {
                "action": {
                    "name": "action name",
                    "args": {
                        "args name": "args value"
                        }
                    },
                "thoughts":
                {
                    "text": "thought",
                    "plan": "plan",
                    "criticism": "criticism",
                    "speak": "当前步骤返回给用户的总结",
                    "reasoning": "",
                },

            }
            """
    try:
        thoughts = response.get("thoughts")
        observation = response.get("observation")
        plan = thoughts.get("plan")
        reasoning = thoughts.get("reasoning")
        criticism = thoughts.get("criticism")
        promt = f"plan:{plan}\nreasoning:{reasoning}\ncriticism:{criticism}\nobservation:{observation}"
        return promt
    except Exception as err:
        print("parse thoughts err:",err)
        return "".format(err)
def agent_excuter(query ,max_request_time = 10):
    cur_request_time = 0
    chat_history = []
    agent_scratch = ""
    while cur_request_time < max_request_time:
        cur_request_time += 1

        prompt = gen_prompt(query, agent_scratch)
        start_time = time.time()
        print("***********{}.开始调用大模型***********".format(cur_request_time),flush=True)
        response = mp.chat(prompt, chat_history)
        end_time = time.time()
        print("***********{}.结束调用大模型，耗时{}.........".format(cur_request_time, end_time - start_time),flush = True)
        if not response or not isinstance(response, dict):
            print("调用大模型失败，即将重试")
            continue
        """
        response: {
            "action": {
                "name": "action name",
                "args": {
                    "args name": "args value"
                    }
                },
            "thoughts":
            {
                "text": "thought",
                "plan": "plan",
                "criticism": "criticism",
                "speak": "当前步骤返回给用户的总结",
                "reasoning": "",
            },
            
        }      
        """
        action_info = response.get("action")
        action_name = action_info.get("name")
        action_args = action_info.get("args")

        print("当前action_name:", action_name,action_args)

        if action_name == "finish":
            final_answer = action_args.get("answer")
            print("final_answer:", final_answer)
            break

        observation = response.get("observation")
        try:
            func = tools_map.get(action_name)
            call_func_result = func(**action_args)

        except Exception as err:
            print("调用大模型失败，即将重试",err)

        agent_scratch = agent_scratch + "\n :observation:{} \n execute action result: {}".format(observation,call_func_result)


        assistant_msg = parser_thoughts(response)
        chat_history.append([user_promt, assistant_msg])
    if cur_request_time == max_request_time:
        print("调用大模型失败，超过最大重试次数")
    else:
        print("调用大模型成功")

def main():
    max_request_time = 10
    while True:
        query = input("请输入你的目标: ")
        if query == "exit":
            return
        agent_excuter(query,max_request_time = max_request_time)

if __name__ == '__main__':
    main()