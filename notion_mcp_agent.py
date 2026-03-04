import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import FunctionCallTermination, TextMentionTermination
load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")

model = OpenAIChatCompletionClient(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    model="sourceful/riverflow-v2-fast",
    model_info={
        "family": "riverflow",
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "multiple_system_messages": True
    },
    request_timeout=120
)

async def config():
    params = StdioServerParams(
        command="npx",
        args=["-y", "@notion-mcp/notion-mcp"],
        env={
            'NOTION_API_KEY': NOTION_API_KEY
        },
    )

    mcp_tools = await mcp_server_tools(server_params=params)

    agent = AssistantAgent(
        name='notion_agent',
        system_message="""You are a helpful assistant that can search and summarize content from the user's notion workspace
        and also list what is asked try to assume the tool and call the same and get the answer. 
        say 'TERMINATE' when you are done with the tools 
        """,
        model_client=model,
        tools=mcp_tools,
        reflect_on_tool_use=True
    )

    team = RoundRobinGroupChat(
        participants=[agent],
        max_turns=5,
        termination_condition=TextMentionTermination('TERMINATE')
    )
    return team

async def orchestrate(team,task):
    async for msg in team.run_stream(task=task):
        yield msg

async def main():
    team = await config()
    task = 'Create a new page titled "PageFromMCPNotion"'

    async for msg in orchestrate(team,task):
        print('-'*100)
        print(msg)
        print('-'*100)

if __name__ == '__main__':
    asyncio.run(main())