import json
import re
import os
from typing import Any, Optional
from openai import OpenAI
import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger
from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
# from prompt_templates import *
# from agent_data_sft import parse_tool_call
import uuid
import pickle
from string import Template
import ast


from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model



def parse_tool_call(tool_call):
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", tool_call, re.DOTALL)
    tool_call_str = matches[-1] if matches else tool_call
    tool_name = tool_call_str[ : tool_call_str.find("(")].strip()
    try:
        tool_args = ast.literal_eval(
            tool_call_str[tool_call_str.find("{") : tool_call_str.rfind("}")+1].strip().replace("\n", "").replace("\\", "")
        )
    except Exception as e:
        logger.info(f"Error parsing tool args: {e}")
        tool_args = {}
    ans = None
    matches = re.findall(r"<answer>(.*?)</answer>", tool_call, re.DOTALL)
    if matches:
        ans = matches[0].strip()
    return {"name": tool_name, "arguments": tool_args, "answer": ans}



def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """

    try:
        response.model = _parse_ft_model_name(
            response.model
        )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    try:
        usage: Optional[Usage] = response.get("usage")
        if usage is None:
            return None
        return {
            "completion_tokens": usage.completion_tokens,
            "prompt_tokens": usage.prompt_tokens,
        }
    except Exception as e:
        logger.error(e)
        return None


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages



def to_aglllm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of AGL messages.
    """
    agl_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            agl_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            agl_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            agl_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            agl_messages.append({"role": "system", "content": message.content})
    return agl_messages




def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    try:
        if os.getenv("OPENAI_API_TYPE", "KEY").lower() == "key":
            response = completion(
                model=model,
                messages=litellm_messages,
                tools=tools,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("OPENAI_API_VERSION"),
                tool_choice=tool_choice,
                **kwargs,
            )
        elif os.getenv("OPENAI_API_TYPE", "KEY").lower() == "vllm":
            response = completion(
                model=model,
                messages=litellm_messages,
                tools=tools,
                api_key=os.getenv("AZURE_OPENAI_VLLM_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_VLLM_ENDPOINT"),
                api_version=os.getenv("OPENAI_API_VERSION"),
                tool_choice=tool_choice,
                **kwargs,
            )
        else:
            scope = os.environ.get("AZURE_BEARER_TOKEN_SCOPE", "api://trapi/.default")
            client_id = os.environ.get("AZURE_CLIENT_ID")
            credential = get_bearer_token_provider(ChainedTokenCredential(
                AzureCliCredential(),
                ManagedIdentityCredential(client_id=client_id)
            ),scope)
            api_version = os.getenv("OPENAI_API_VERSION")
            # deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
            response = completion(
                model=model,
                messages=litellm_messages,
                tools=tools,
                api_base=endpoint,
                api_version=api_version,
                azure_ad_token_provider=credential,
                tool_choice=tool_choice,
                **kwargs,
            )
    except Exception as e:
        logger.error(e)
        raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message


def agl_tc_generate(
    model: str,
    llm_endpoint: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        llm_endpoint: The endpoint of the model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES


    client = OpenAI(
        base_url=llm_endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
    )
    logger.info(f"AGL Generate using model: {model} at endpoint: {llm_endpoint}, key: {os.environ.get('OPENAI_API_KEY', 'token-abc123')[-6:]}")
    
    agl_messages = to_aglllm_messages(messages)
    

    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=agl_messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
            tools=tools,
            tool_choice=tool_choice,
        )

    except Exception as e:
        logger.error(e)
        raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    
    agl_tool_calls = []
    for tool_call in tool_calls:
        try:
            arguments_str = tool_call.function.arguments
            arguments_str = arguments_str.replace('\n', '').replace('\\', '')
            arguments_str = arguments_str[arguments_str.find("{") : arguments_str.rfind("}")+1].strip()
            arguments = ast.literal_eval(arguments_str)
        except Exception as e:
            logger.info(f"Error parsing tool args: {e}, arguments_str: {arguments_str}")
            arguments = {}
        agl_tool_calls.append(
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=arguments,
            )
        )

    agl_tool_calls = agl_tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=agl_tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message




def agl_generate(
    model: str,
    llm_endpoint: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        llm_endpoint: The endpoint of the model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    with open(f"/mnt/storage/data/tau/prompts/all_prompts.pkl", "rb") as f:
        all_prompts = pickle.load(f)
        
    client = OpenAI(
        base_url=llm_endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
    )
    logger.info(f"AGL Generate using model: {model} at endpoint: {llm_endpoint}, key: {os.environ.get('OPENAI_API_KEY', 'token-abc123')[-6:]}")
    
    retrieved_context_list = []
    for message in messages:
        # logger.info(f"Processing message: {message}")
        if isinstance(message, AssistantMessage) and message.is_tool_call():
            tool_calls = None
            for tc in message.tool_calls:
                retrieved_context_list.append(f"<tool_call>{tc.name}({tc.arguments})</tool_call>")
        elif isinstance(message, ToolMessage):
            retrieved_context_list.append(f"<tool_response>{message.content}</tool_response>")
        else:
            retrieved_context_list.append(f"{message.role}: {message.content}")
        
    retrieved_context = "\n".join(retrieved_context_list)
    prompt = all_prompts["tool_calling_template"].substitute(
        existing_context=retrieved_context,
        available_tools=all_prompts["tool_calling_prompt"],
        instructions=all_prompts["tool_calling_instructions"],
    )
    agl_messages=[{"role": "system", "content": all_prompts["tool_calling_system_prompt"]}, {"role": "user", "content": prompt}]

    try:
        ori_response = client.chat.completions.create(
            model=model,
            messages=agl_messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        response = ori_response.choices[0].message.content
        # logger.info(f"AGL Tool Call Prompt: {prompt}")
        # logger.info(f"AGL Tool Call Response: {response}")

        cost = get_response_cost(ori_response)
        usage = get_response_usage(ori_response)

        resp = parse_tool_call(response)
        tool_name = resp["name"]
        function_tool_dict = all_prompts["function_tool_dict"]
        tool_calls = []
        if tool_name in function_tool_dict:
            content = f"Decide to call tool {tool_name}."
            tool_calls = [
                ToolCall(
                    id=str(uuid.uuid1()),
                    name=resp["name"],
                    arguments=resp["arguments"],
                )
            ]
        elif tool_name == all_prompts["termination_tool"]:
            content = "Decide to stop tool calling and return to the user."
        else:
            content = f"Tool {tool_name} not found. Valid tools are: {list(function_tool_dict.keys())}. The correct tool calling format shall be: <tool_call> <<placeholder for the called tool name>> ({{\"parameter\": \"value\", ...}}) </tool_call>."
            logger.info(content)
    except Exception as e:
        logger.error(e)
        tool_calls = None
        content = "Error occurred during tool call parsing."
        cost = 0.0
        usage = None
        # raise e
   
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data={},
    )
    return message




def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
