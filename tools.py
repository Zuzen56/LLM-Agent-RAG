import os
from langchain_community.tools.tavily_search import TavilySearchResults
import json

def _get_workdir_root():
    workdir_root = os.environ.get("WORKDIR_ROOT","./data/llm_result")
    return workdir_root


WORKDIR_ROOT = _get_workdir_root()

def read_file(filename):
    filename = os.path.join(WORKDIR_ROOT,filename)

    if not os.path.exists(filename):
        return f"file {filename} not exists"
    with open(filename, "r") as f:
        return "\n".join(f.readlines())


def append_to_file(filename, content):
    filename = os.path.join(WORKDIR_ROOT,filename)
    if not os.path.exists(filename):
        return f"file {filename} not exists"

    with open(filename,"a") as f:
        f.write(content)

    return "append content to file success"

def write_to_file(filename, content):
    filename = os.path.join(WORKDIR_ROOT,filename)
    if not os.path.exists(WORKDIR_ROOT):
        os.makedirs(WORKDIR_ROOT)

    with open(filename, "w") as f:
        f.write(content)

    return "write content to file success"

def search(query):
    tavily = TavilySearchResults(max_results = 5)

    try:
        ret = tavily.invoke(input=query)

        """
        ret:
        [{
            "url": ,
            "content": "",
        }]
        """
        content_list = [obj["content"] for obj in ret]
        return "\n".join(content_list)

    except Exception as err:
        return "search failed: {}".format(err)

tools_info = [
    {
        "name": "read_file",
        "description": "read file content from agent genrator,should write file before read",
        "args": [
            {
                "name": "filename",
                "type": "string",
                "description": "read file name"
            }
        ]
    },

    {
        "name": "write_to_file",
        "description": "write llm content to file",
        "args": [{
                "name": "filename",
                "type": "string",
                "description": "read file name"
            },
            {
                "name": "content",
                "type": "string",
                "description": "write to file content"
            }

        ]
    },

    {
        "name": "append_to_file",
        "description": "append llm content to file,should write file before read",
        "args": [{
            "name": "filename",
            "type": "string",
            "description": "read file name"
        },
            {
                "name": "content",
                "type": "string",
                "description": "append to file content"
            }

        ]
    },

    {
        "name": "search",
        "description": "this is a search engine,you can gain additional konwledge by this tool when you are confused",
        "args": [{
            "name": "query",
            "type": "string",
            "description": "search query to look up"
        }]
    }
]


tools_map = {
    "read_file": read_file,
    "write_to_file": write_to_file,
    "append_to_file": append_to_file,
    "search": search
}

def gen_tools_desc():
    tools_desc = []
    for idx,t in enumerate(tools_info):
        args_desc = []
        for info in t["args"]:
            args_desc.append({
                "name": info["name"],
                "type": info["type"],
                "description": info["description"]
            })
        args_desc = json.dumps(args_desc, ensure_ascii=False)
        tools_desc = f"{idx+1}. {t['name']}: {t['description']}, args: {args_desc}"
        tools_desc.append(tools_desc)
    tools_prompt = "\n".join(tools_desc)
    return tools_prompt



