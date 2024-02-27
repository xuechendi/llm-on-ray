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
import os

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks import StreamingStdOutCallbackHandler

parser = argparse.ArgumentParser(
    description="Example script of enable langchain agent", add_help=True
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

# =================================================#

tools = [DuckDuckGoSearchRun(max_results=1)]
prompt = ChatPromptTemplate.from_messages([("human", "{input}")])

model = ChatOpenAI(
    openai_api_base=openai_base_url,
    model_name=args.model_name,
    openai_api_key=openai_api_key,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind_tool(tools)

runnable = prompt | model

runnable.invoke({"input": "what is the weather in sf"})
