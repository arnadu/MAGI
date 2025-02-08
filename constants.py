from enum import Enum
from dataclasses import dataclass
from typing import List, Union, Annotated
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

#names of the providers of the LLMs; they will appear in dropdown
# will also be used as keys in api_key_state dictionary and in the browser's local storage
class LLMProvider:
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LAMBDALABS = "lambdalabs"
    HUGGINGFACE = "huggingface"

#names of the system environment variables that store the API keys of these providers
ENV_VAR_NAMES = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.LAMBDALABS: "LAMBDALABS_API_KEY",
    LLMProvider.HUGGINGFACE: "HUGGINGFACE_API_KEY"
}

#OPENAI models
#LLM_MODEL = "gpt-3.5-turbo-0125"
#LLM_MODEL = "gpt-4-1106-preview"
#LLM_MODEL = "gpt-4-0125-preview"
#LLM_MODEL = "gpt-4-turbo-2024-04-09"
#LLM_MODEL = "gpt-4o-2024-05-13"
#LLM_MODEL = "gpt-4o-mini-2024-07-18"
OAI_GPT_4o = "gpt-4o-2024-08-06"
OAI_GPT_o1 = "o1-2024-12-17"


#Anthropic models
A_CLAUDE_3_OPUS = "claude-3-opus-20240229"  
A_CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"

#LAMBDALABS models
LL_LLAMA_3_3_70B_INSTRUCT = "llama3.3-70b-instruct-fp8" # $0.20 per 1 million token

#HUGGINGFACE models
HF_LLAMA_3_3_70B_INSTRUCT = "meta-llama/Llama-3.3-70B-Instruct"

TEMPERATURE = 0.0

@dataclass
class LLMModel:
    api_key: str = ''
    provider: str = LLMProvider.OPENAI
    model: str = OAI_GPT_4o
    temperature = 0

#JSON schema returned by the LLM
LLM_NEXT_ACTION_PROCESS_INFORMATION = "process_information"
LLM_NEXT_ACTION_ASK_USER = "ask_user"

class NextAction(str, Enum):
    process_information = LLM_NEXT_ACTION_PROCESS_INFORMATION
    ask_user = LLM_NEXT_ACTION_ASK_USER 

class LLMAnswer(BaseModel):
    next_action: Annotated[NextAction, Field(description="whether you need to ask the user something or continue processing the information you have just obtained")]
    msg_to_user: str = Field(description="your messgage to the user, either to let them know what you are doing or to ask them something")


#schema returned by tool calls as well as call_llm_tool
MSG_SOURCE_SYSTEM = "system" 
MSG_SOURCE_ASSISTANT = "assistant" #final answer from the assistant to the user
MSG_SOURCE_EDITOR = "EDITOR"  #this is the source of the message when the Editor is called
MSG_SOURCE_QUERY = "LIBRARIAN"  #this is the source of the message when the Librarian tool is called
MSG_SOURCE_ANALYZE = "CRITIQUE" #this is the source of the message when the Critique tool is called

@dataclass
class AssistantAnswer():
    msg_source: str = MSG_SOURCE_ASSISTANT  #assistant | tool name
    tool_input: str = "" #the input to the tool when the message is a tool call
    next_action: str = LLM_NEXT_ACTION_PROCESS_INFORMATION  #returned by LLM to decide whether to ask the user something or continue processing the information
    msg_to_user: str = "" #the message to the user or the result of the tool call
