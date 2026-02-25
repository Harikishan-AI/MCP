import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import FunctionCallTermination, TextMentionTermination
load_dotenv()

model = OpenAIChatCompletionClient(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    model="arcee-ai/trinity-large-preview:free",
    model_info={
        "family": "arcee",
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "multiple_system_messages": True
    },
)

NOTION_TOKEN = 