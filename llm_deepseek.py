from typing import Iterable, Iterator, Literal, override, cast

import httpx
import llm
from llm.parts import StreamEvent, TextPart
from llm.utils import remove_dict_none_values, simplify_usage_dict
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


API_BASE_URL = 'https://api.deepseek.com'


class DeepSeekModel(llm.KeyModel):
    needs_key = 'deepseek'
    key_env_var = 'DEEPSEEK_API_KEY'
    can_stream = True
    supports_schema = False # TODO
    supports_tools = False # TODO

    class Options(llm.KeyModel.Options):
        reasoning_effort: Literal['high', 'max'] | None = None
        max_tokens: int | None = None
        stop: str | None = None
        temperature: float | None = None  # [0, 2], default to 1
        top_p: float | None = None  # [0, 1], default to 1
        user_id: str | None = None

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
            for chunk in completion:
                for choice in chunk.choices:
                    delta = choice.delta
                    delta_reasoning_content = getattr(delta, 'reasoning_content', None)
                    if delta_reasoning_content:
                        yield StreamEvent(type='reasoning', chunk=delta_reasoning_content)
                    if delta.content:
                        yield StreamEvent(type='text', chunk=delta.content)
                if chunk.usage:
                    usage = chunk.usage
        else:
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                stream=False,
                **kwargs,
            )
            for choice in completion.choices:
                message = choice.message
                reasoning_content = getattr(message, 'reasoning_content', None)
                if reasoning_content:
                    yield StreamEvent(type='reasoning', chunk=reasoning_content)
                if message.content:
                    yield StreamEvent(type='text', chunk=message.content)
            if completion.usage:
                usage = completion.usage
        if usage:
            response.set_usage(input=usage.prompt_tokens, output=usage.completion_tokens, details=simplify_usage_dict({
                'cached_tokens': usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else None,
                'reasoning_tokens': usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else None,
            }))

    def build_messages(self, prompt: llm.Prompt, conversation: llm.Conversation | None) -> Iterable[ChatCompletionMessageParam]:
        messages = []
        for message in prompt.messages:
            texts = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    texts.append(part.text)
            if not texts and message.role in ('user', 'system'):
                continue
            messages.append({'role': message.role, 'content': ''.join(texts)})
        return messages


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
