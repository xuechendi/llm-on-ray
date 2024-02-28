#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
from openai import OpenAI
import os

parser = argparse.ArgumentParser(
    description="Example script to query with openai sdk", add_help=True
)
parser.add_argument(
    "--model_name",
    default="mistral-7b-instruct-v0.2",
    type=str,
    help="The name of model to request",
)
parser.add_argument(
    "--streaming_response",
    default=False,
    action="store_true",
    help="Whether to enable streaming response",
)
parser.add_argument(
    "--max_new_tokens", default=512, help="The maximum numbers of tokens to generate"
)
args = parser.parse_args()

if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.environ["OPENAI_API_KEY"]
else:
    openai_api_key = "not_needed"

if "OPENAI_BASE_URL" in os.environ:
    openai_base_url = os.environ["OPENAI_BASE_URL"]
elif openai_api_key == "not_needed":
    openai_base_url = "http://localhost:8000/v1"
else:
    openai_base_url = "https://api.openai.com/v1"

# ================================================ #
# Lets define a function/tool for getting the weather. In this demo it we mockthe output
# In real life, you'd end up calling a library/API such as PWOWM (open weather map) library:
# Depending on your app's functionality, you may also, call vendor/external or internal custom APIs


def get_current_weather(location, unit):
    # Call an external API to get relevant information (like serpapi, etc)
    # Here for the demo we will send a mock response
    weather_info = {
        "location": location,
        "temperature": "78",
        "unit": unit,
        "forecast": ["sunny", "with a chance of meatballs"],
    }
    return weather_info


# ================================================ #

client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What's the weather like in Boston today?"},
]

completion = client.chat.completions.create(
    model=args.model_name,
    messages=messages,
    max_tokens=args.max_new_tokens,
    tools=tools,
    tool_choice="auto",
    stream=args.streaming_response,
)

if args.streaming_response:
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
else:
    try:
        content = completion.choices[0].message.content
        if content is not None:
            print(content, end="", flush=True)
    except Exception:
        print(content)

print("")
