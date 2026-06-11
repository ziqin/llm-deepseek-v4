import json
from typing import Iterable, Iterator, Literal, override, cast

import httpx
import llm
from llm.parts import ReasoningPart, StreamEvent, TextPart, ToolCallPart, ToolResultPart
from llm.utils import remove_dict_none_values, simplify_usage_dict
from openai import OpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionFunctionToolParam, ChatCompletionMessageFunctionToolCallParam, ChatCompletionMessageParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.shared_params import FunctionDefinition


API_BASE_URL = 'https://api.deepseek.com'


class DeepSeekModel(llm.KeyModel):
    needs_key = 'deepseek'
    key_env_var = 'DEEPSEEK_API_KEY'
    can_stream = True
    supports_schema = False
    supports_tools = True

    class Options(llm.KeyModel.Options):
        reasoning_effort: Literal['high', 'max'] | None = None
        max_tokens: int | None = None
        stop: str | None = None
        temperature: float | None = None  # [0, 2], default to 1
        top_p: float | None = None  # [0, 1], default to 1
        user_id: str | None = None
        json_output: bool = False

    @override
    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id

    @override
    def execute(self, prompt: llm.Prompt, stream: bool, response: llm.Response,
                conversation: llm.Conversation | None, key: str | None) -> Iterator[StreamEvent]:
        options = cast(DeepSeekModel.Options, prompt.options)
        messages = self.build_messages(prompt, conversation)
        client = OpenAI(api_key=self.get_key(key), base_url=API_BASE_URL)
        thinking_mode = options.reasoning_effort is not None
        kwargs = remove_dict_none_values({
            'reasoning_effort': options.reasoning_effort,
            'max_tokens': options.max_tokens,
            'stop': options.stop,
            'temperature': options.temperature if not thinking_mode else None,
            'top_p': options.top_p if not thinking_mode else None,
            'tools': self._build_tools(prompt.tools) if prompt.tools else None,
            'response_format': {'type': 'json_object' if options.json_output else 'text'},
            'extra_body': {
                'thinking': {'type': 'enabled' if thinking_mode else 'disabled'},
                'user_id': options.user_id,
            },
        })
        usage = None
        if stream:
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                stream=True,
                stream_options={'include_usage': True},
                **kwargs,
            )
            tool_calls: dict[int, ChoiceDeltaToolCall] = {}
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    delta_reasoning_content = getattr(delta, 'reasoning_content', None)
                    if delta_reasoning_content:
                        yield StreamEvent(type='reasoning', chunk=delta_reasoning_content)
                    for tool_call in delta.tool_calls or []:
                        if tool_call.function is None:
                            continue
                        if tool_call.function.arguments is None:
                            tool_call.function.arguments = ''
                        idx = tool_call.index
                        if idx not in tool_calls:
                            tool_calls[idx] = tool_call
                            yield StreamEvent(type='tool_call_name', chunk=tool_call.function.name or '', tool_call_id=tool_call.id)
                        else:
                            tool_calls[idx].function.arguments += tool_call.function.arguments  # type: ignore[union-attr]
                        if tool_call.function.arguments:
                            yield StreamEvent(type='tool_call_args', chunk=tool_call.function.arguments, tool_call_id=tool_calls[idx].id)
                    if delta.content:
                        yield StreamEvent(type='text', chunk=delta.content)
                if chunk.usage:
                    usage = chunk.usage
            for tool_call in tool_calls.values():
                if tool_call.function is None:
                    continue
                response.add_tool_call(llm.ToolCall(
                    tool_call_id=tool_call.id,
                    name=cast(str, tool_call.function.name),
                    arguments=json.loads(cast(str, tool_call.function.arguments)),
                ))
        else:
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                stream=False,
                **kwargs,
            )
            if completion.choices:
                choice = completion.choices[0]
                message = choice.message
                reasoning_content = getattr(message, 'reasoning_content', None)
                if reasoning_content:
                    yield StreamEvent(type='reasoning', chunk=reasoning_content)
                for tool_call in message.tool_calls or []:
                    if isinstance(tool_call, ChatCompletionMessageToolCall):
                        yield StreamEvent(type='tool_call_name', chunk=tool_call.function.name, tool_call_id=tool_call.id)
                        yield StreamEvent(type='tool_call_args', chunk=tool_call.function.arguments, tool_call_id=tool_call.id)
                        response.add_tool_call(llm.ToolCall(
                            tool_call_id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=json.loads(tool_call.function.arguments),
                        ))
                if message.content:
                    yield StreamEvent(type='text', chunk=message.content)
            if completion.usage:
                usage = completion.usage
        if usage:
            response.set_usage(input=usage.prompt_tokens, output=usage.completion_tokens, details=simplify_usage_dict({
                'cached_tokens': usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else None,
                'reasoning_tokens': usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else None,
            }))

    @staticmethod
    def _build_tools(tools: list[llm.Tool]) -> Iterable[ChatCompletionFunctionToolParam]:
        for tool in tools:
            definition: FunctionDefinition = {'name': tool.name, 'parameters': tool.input_schema}
            if tool.description is not None:
                definition['description'] = tool.description
            yield {'type': 'function', 'function': definition}

    @staticmethod
    def build_messages(prompt: llm.Prompt, conversation: llm.Conversation | None) -> Iterable[ChatCompletionMessageParam]:
        for message in prompt.messages:
            if message.role == 'system':
                content = ''.join(part.text for part in message.parts if isinstance(part, TextPart))
                if content:
                    yield {'role': 'system', 'content': content}
            elif message.role == 'user':
                content = ''.join(part.text for part in message.parts if isinstance(part, TextPart))
                if content:
                    yield {'role': 'user', 'content': content}
            elif message.role == 'assistant':
                content = ''.join(part.text for part in message.parts if isinstance(part, TextPart))
                reasoning = ''.join(part.text for part in message.parts if isinstance(part, ReasoningPart))
                tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = [
                    {
                        'type': 'function',
                        'id': cast(str, part.tool_call_id),
                        'function': {'name': part.name, 'arguments': json.dumps(part.arguments)},
                    }
                    for part in message.parts if isinstance(part, ToolCallPart)
                ]
                assistant_message: ChatCompletionAssistantMessageParam = {'role': 'assistant', 'content': content or None}
                if reasoning:
                    assistant_message['reasoning_content'] = reasoning  # type: ignore[typeddict-unknown-key]
                if tool_calls:
                    assistant_message['tool_calls'] = tool_calls
                yield assistant_message
            elif message.role == 'tool':
                for part in message.parts:
                    if isinstance(part, ToolResultPart):
                        yield {
                            'role': 'tool',
                            'tool_call_id': cast(str, part.tool_call_id),
                            'content': part.output,
                        }


@llm.hookimpl
def register_models(register):
    register(DeepSeekModel(model_id='deepseek-v4-flash'))
    register(DeepSeekModel(model_id='deepseek-v4-pro'))


class BearerAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request):
        request.headers['Authorization'] = 'Bearer ' + self.token
        yield request


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def deepseek():
        '''Commands for working directly with DeepSeek API'''

    @deepseek.command()
    def balance():
        '''Display balance of the DeepSeek account'''
        key = llm.get_key(alias=DeepSeekModel.needs_key, env=DeepSeekModel.key_env_var)
        if key is None:
            raise llm.NeedsKeyException(f"No key found - add one using 'llm keys set {DeepSeekModel.needs_key}' or set the {DeepSeekModel.key_env_var} environment variable")
        resp = httpx.get(API_BASE_URL + '/user/balance', auth=BearerAuth(key))
        resp.raise_for_status()
        resp_body = resp.json()
        for index, info in enumerate(resp_body['balance_infos']):
            if index != 0:
                print()
            currency = info['currency']
            print(f'Total available balance: {info['total_balance']:>10} {currency}')
            print(f'Granted balance:         {info['granted_balance']:>10} {currency}')
            print(f'Topped-up balance:       {info['topped_up_balance']:>10} {currency}')
