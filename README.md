# llm-deepseek-v4

An [LLM](https://llm.datasette.io/) plugin to access [DeepSeek](https://www.deepseek.com/)'s V4 models.

This plugin uses the LLM v0.32+ `StreamEvent` API to support thinking mode. Chain of thoughts are written to stderr, displayed with a dim color, and recorded in logs.

## Installation

```shell
git clone https://github.com/ziqin/llm-deepseek-v4.git
cd llm-deepseek-v4
llm install --editable .
```

## Usage

Configure the model by setting a key called `deepseek` to your [API key](https://platform.deepseek.com/api_keys):
```shell
llm keys set deepseek
```

You can also set the API key by assigning it to the environment variable `DEEPSEEK_API_KEY`.

Now run the model using `-m deepseek-v4-flash` or `-m deepseek-v4-pro`, for example:
```shell
llm -m deepseek-v4-flash "Hi"
```

To enable thinking mode, set the `reasoning_effort` option to `high` or `max`, for example:
```shell
llm -m deepseek-v4-flash -o reasoning_effort high "I need to get my car washed. The car wash is 100m away. Should I go by car or by foot?"
```

Display account balance:
```shell
llm deepseek balance
```

## Available models

* deepseek-v4-flash
* deepseek-v4-pro
