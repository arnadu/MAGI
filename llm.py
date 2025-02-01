from datetime import datetime
from dataclasses import dataclass
import random
import time
import json
import os
import textwrap

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

from tenacity import retry, wait_random_exponential, stop_after_attempt, after_log

import constants as c

import openai
from litellm import completion as litellm_completion


@retry(wait=wait_random_exponential(multiplier=1, max=10), stop=stop_after_attempt(3), after=after_log(logger, logging.DEBUG), reraise=True) 
def call_llm_once(llm, messages, tooling_descriptions=None, response_format=None):
    """make a call to the LLM chat completion API; retries up to 3 times with exponential backoff if the call fails"""

    match llm.provider: 
    
        case c.LLMProvider.HUGGINGFACE: #use HF' inference interface  WORK IN PROGRESS - THIS IS NOT WORKING AS EXPECTED

            hf_endpoint = "https://api-inference.huggingface.co/models/" #https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions
            hf_llama_model = llm.model #"meta-llama/Llama-3.3-70B-Instruct"
            llm_client = openai.Client(
                base_url=f"{hf_endpoint}{hf_llama_model}/v1/",
                api_key=llm.api_key
            )

            params = {'messages': messages, 'model': llm.model, 'temperature': llm.temperature}

            if response_format:
                if response_format == "json_object":
                    params['response_format'] = {"type": "json_object"}
                else:
                    json_model = response_format.model_json_schema()
                    #json_model = { #simplified version
                    #    "type": "json",
                    #    "value": {
                    #        "properties": {
                    #            "next_action": {"enum": [c.LLM_NEXT_ACTION_PROCESS_INFORMATION, c.LLM_NEXT_ACTION_ASK_USER]},
                    #            "msg_to_user": {"type": "string"},
                    #        },
                    #        "required": ["next_action", "msg_to_user"],
                    #    },
                    #}
                    if tooling_descriptions:
                        json_prompt = "**Output:**Format your response as a JSON object according to the following schema. Do not include any other text in your response:```\n" + json.dumps(json_model) + "\n```" #json_model + "\n```"
                        messages[0]['content'] += json.dumps(json_model)
                    else: #grammar and tools are mutually incompatible in HuggingFace's endpoint
                        params['response_format'] = response_format

            if tooling_descriptions:
                params['tools'] = tooling_descriptions
                params['tool_choice'] = "auto"

            chat_completion = llm_client.chat.completions.create(**params)

            #chat_completion.choices[0].message.parsed = None
            #if chat_completion.choices[0].message.content is not None:
            #    if response_format:
            #        if response_format == "json_object": 
            #            chat_completion.choices[0].message.parsed = json.loads(chat_completion.choices[0].message.content)
            #        else:
            #            chat_completion.choices[0].message.parsed = response_format.model_validate_json(chat_completion.choices[0].message.content)

            #make sure that there is a content field in the message

            #fudge to make follow-up llm calls work... there is a bug in the SDK see https://github.com/huggingface/text-generation-inference/issues/2461
            message = chat_completion.choices[0].message
            if message.tool_calls:
                if not message.content:
                    message.content=''  #HUGGINGFACE expects a content field otherwise it fails
                #for i in range(len(message.tool_calls)):
                #    message.tool_calls[i] = json.dumps(message.tool_calls[i].function.arguments) 

            return chat_completion  

        case c.LLMProvider.LAMBDALABS: #WORK IN PROGRESS - THIS IS NOT WORKING AS EXPECTED

            lambdalabs_api_base = "https://api.lambdalabs.com/v1"
            llm_client = openai.OpenAI(api_key=llm.api_key, base_url=lambdalabs_api_base)

            params = {'messages': messages, 'model': llm.model, 'temperature': llm.temperature}
            if response_format:
                if response_format == "json_object":
                    params['response_format'] = {"type": "json_object"}
                else:
                    json_model = json.dumps(response_format.model_json_schema())
                    json_prompt = "**Output:**Format your response as a JSON object according to the following schema. Do not include any other text in your response:```\n" + json_model + "\n```"
                    messages[0]['content'] += json_prompt
                    #params['response_format'] = {'type': 'json_object'}
            if tooling_descriptions:
                params['tools'] = tooling_descriptions
                params['tool_choice'] = "auto"
            
            completion = llm_client.chat.completions.create(**params)
            
            #if response_format:
            #    if response_format == "json_object":
            #        completion['parsed'] = json.loads(completion.choices[0].message.content)
            #    else:
            #        completion['parsed'] = response_format.model_validate_json(completion.choices[0].message.content)
                         
            return completion

        case c.LLMProvider.ANTHROPIC:

            params = {'messages': messages, 'model': llm.model, 'temperature': llm.temperature}

            params['api_key'] = llm.api_key 

            if response_format == "json_object": #json_object is defined by OPENAI's API
                params['response_format'] = {"type": "json_object"}
            else:
                params['response_format'] = response_format

            if tooling_descriptions:
                params['tools'] = tooling_descriptions
                params['tool_choice'] = "auto"

            response = litellm_completion(**params)
            print(response)            
            return response

        case _: #default to OPENAI interface         

            llm_client = openai.OpenAI(api_key=llm.api_key)
            params = {'messages':messages, 'model':llm.model, 'temperature':llm.temperature}

            if "o1" in llm.model:
                params.pop("temperature") #o1 models do not support temperature
                params["reasoning_effort"] = "medium" #o1 models have a new parameter called reasoning_effort
                for m in params['messages']: #o1 models do not support system messages, but they do support developer messages
                    if m['role'] == 'system':
                        m['role'] = 'developer'
           
            if response_format: #response_format can be either None, "json_object" or LLMAnswer
                if response_format == "json_object": #json_object is defined by OPENAI's API
                    params['response_format'] = {"type": "json_object"}
                else:
                    params['response_format'] = response_format

            if tooling_descriptions:
                params['tools'] = tooling_descriptions
                params['tool_choice'] = "auto"

            chat_completion = llm_client.beta.chat.completions.parse(**params)
            
            return chat_completion


 

