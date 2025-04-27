import asyncio
import json
import os
from typing import List, Optional
from contextlib import AsyncExitStack
import warnings

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=ResourceWarning)

def clean_schema(schema): # Cleans the schema by keeping only allowed keys
    allowed_keys = {"type", "properties", "required", "description", "title", "default", "enum"}
    return {k: v for k, v in schema.items() if k in allowed_keys}

class MCPGeminiAgent:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-2.0-flash"
        self.tools = None
        self.server_params = None
        self.server_name = None

    async def select_server(self):
        with open('mcp.json', 'r') as f:
            mcp_config = json.load(f)
        servers = mcp_config['mcpServers']
        server_names = list(servers.keys())
        print("Available MCP servers:")
        for idx, name in enumerate(server_names):
            print(f"  {idx+1}. {name}")
        while True:
            try:
                choice = int(input(f"Please select a server by number [1-{len(server_names)}]: "))
                if 1 <= choice <= len(server_names):
                    break
                else:
                    print("That number is not valid. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        self.server_name = server_names[choice-1]
        server_cfg = servers[self.server_name]
        command = server_cfg['command']
        args = server_cfg.get('args', [])
        env = server_cfg.get('env', None)
        self.server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )

    async def connect(self):
        await self.select_server()
        self.stdio_transport = await self.exit_stack.enter_async_context(stdio_client(self.server_params))
        self.stdio, self.write = self.stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        print(f"Successfully connected to: {self.server_name}")
        # List available tools for this server
        mcp_tools = await self.session.list_tools()
        print("\nAvailable MCP tools for this server:")
        for tool in mcp_tools.tools:
            print(f"- {tool.name}: {tool.description}")

    async def agent_loop(self, prompt: str) -> str:
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        mcp_tools = await self.session.list_tools()
        tools = types.Tool(function_declarations=[
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": clean_schema(getattr(tool, "inputSchema", {}))
            }
            for tool in mcp_tools.tools
        ])
        self.tools = tools
        response = await self.genai_client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                tools=[tools],
            ),
        )
        contents.append(response.candidates[0].content)
        turn_count = 0
        max_tool_turns = 5
        while response.function_calls and turn_count < max_tool_turns:
            turn_count += 1
            tool_response_parts: List[types.Part] = []
            for fc_part in response.function_calls:
                tool_name = fc_part.name
                args = fc_part.args or {}
                print(f"Invoking MCP tool '{tool_name}' with arguments: {args}")
                tool_response: dict
                try:
                    tool_result = await self.session.call_tool(tool_name, args)
                    print(f"Tool '{tool_name}' executed.")
                    if tool_result.isError:
                        tool_response = {"error": tool_result.content[0].text}
                    else:
                        tool_response = {"result": tool_result.content[0].text}
                except Exception as e:
                    tool_response = {"error":  f"Tool execution failed: {type(e).__name__}: {e}"}
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name, response=tool_response
                    )
                )
            contents.append(types.Content(role="user", parts=tool_response_parts))
            print(f"Added {len(tool_response_parts)} tool response(s) to the conversation.")
            print("Requesting updated response from Gemini...")
            response = await self.genai_client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    tools=[tools],
                ),
            )
            contents.append(response.candidates[0].content)
        if turn_count >= max_tool_turns and response.function_calls:
            print(f"Stopped after {max_tool_turns} tool calls to avoid infinite loops.")
        print("All tool calls complete. Displaying Gemini's final response.")
        return response

    async def chat(self):
        print(f"\nMCP-Gemini Assistant is ready and connected to: {self.server_name}")
        print("Enter your question below, or type 'quit' to exit.")
        while True:
            try:
                query = input("\nYour query: ").strip()
                if query.lower() == 'quit':
                    print("Session ended. Goodbye!")
                    break
                print(f"Processing your request...")
                res = await self.agent_loop(query)
                print("\nGemini's answer:")
                print(res.text)
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    agent = MCPGeminiAgent()
    try:
        await agent.connect()
        await agent.chat()
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    import sys
    import os
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Session interrupted. Goodbye!")
    finally:
        sys.stderr = open(os.devnull, "w") 
