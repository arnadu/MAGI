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

logger.debug("Starting the agent module")

from bs4 import BeautifulSoup

from enum import Enum
from typing import List, Union, Annotated
from pydantic import BaseModel, Field
from utils import get_openai_schema

from library import Library
from database import get_list_of_assessments, save_assessment_todb, load_assessment_fromdb
from database import get_template_fromdb, save_template_todb

import constants as c

from llm import call_llm_once

system_prompt = """You are a useful assistant. You best help the user by adhering to the following principles:
1- Determine the user's context and intent, and write down the information in your ASSISTANT NOTES document;
2- Determine what actions you can take to best help, and write down "Instructions" in your ASSISTANT NOTES document with the Editor tool;
3- Carry out these Instructions to the best of your ability, and write down the results in your ASSISTANT NOTES document; use the Librarian tool to perform research.
5- Continuously interact with the user to gather information, explain your plan and provide explanations, and to review your progress and confirm their satisfaction.
6- Always update the assistant notes with the latest information and the results of your actions with the Editor tool. Do not delay.
7- Regularly re-assess with the Critique tool your interpretation of the user's intent, the actions you can take and your progress in executing these actions, and revise your plan accordingly.
"""

initial_prompt = "Hello, I am your assistant. I would like to know more about you so that I can assist in anything you need. What is your name?"

assessment_template = """
<html>
<head>
<title>ASSISTANT NOTES</title>
<style>
.Instructions {
  background-color: WhiteSmoke;
  color: black;
  font-style: italic;
}
</style>
</head>
<body>
<p class="Instructions">The assistant must edit this document as may be necessary to help the user. 
The assistant should use a hierarchical outline for this document and create sections as it sees fit. In each section, the assistant should record the user's context and intent, the actions it plans to take to help the user, the outcome of these actions, and a list of children section for more elementary objectives.
Elements with class=Instructions should be created as the user's intent becomes clear and the assistant determines what actions it can take; Instructions can be created anywhere in the document</p>
<h1 id="1">Overall Context and Purpose</h1>
<div class="Instructions" id="2">Obtain general information about the user in the context of their professional or personal life</div>
<div id="2">This is an example of a paragraph that the assistant can edit; it is also possible to create siblings to this element</div>
<h1 id="3">Outcome</h1>
<div class="Instructions" id="4">Identify how to break down the user's purpose into achievable tasks and create sub-sections accordingly</div>
<div id="5">This is an example of a section that the assistant can edit; it is also possible to create siblings to this element</div>
</body>
</html>"""

analyze_system_prompt = '''
Review the notes you have taken so far, focusing on the question you are being asked.
Provide constructive criticism and suggestions for changes. In particular you should analysis the following aspects and suggest changes where necessary:
1) have you correctly identified the relevant elements of the user's background?
2) have you understood the users' intent correctly?
3) have you devised practical solutions to help achieve the user by breaking down the problem into achievable tasks?
4) have you written down operational Instructions to help you carry out these tasks?
5) have you carried out these Instructions to the best of your ability?
6) have you written down the results of [partially] conpleted tasks in your notes?
7) are the obtained results actually useful to achieve the user's intent or should you rethink part of the plan?
'''

library_prompt = '''
You are a useful librarian and you help the user research a question. 
A similarity search of the library based on the user's question has returned the following excerpts. 
Now answer the question based on these excerpts. Provide citations of the source text(s) as relevant.  
If it is probable that the excerpts are lacking all the necessary information (for example because necessary definitions were not returned by the similarity search):  answer the question based on available information but also suggest to the user ways they can change the question and broaden their query.
If there is no relevant information in these excerpts: reply so.
'''

system_prompt= textwrap.dedent(system_prompt).strip()
initial_prompt = textwrap.dedent(initial_prompt).strip()
assessment_template = textwrap.dedent(assessment_template).strip()
analyze_system_prompt = textwrap.dedent(analyze_system_prompt).strip()
library_prompt = textwrap.dedent(library_prompt).strip()

#TODO: it would be nice to abstract function names, parameters and return values  [code] <--> [LLM schema] so that one can change the schema without changing the code; this would be useful to fine-tune the prompt engineering

class ApplicationTemplate:
    def __init__(self, app_name=None, revision=None):
        if app_name: #load the template from the database   
            self.load_from_db(app_name, revision)
            
        else: #create a default template
            self.app_class = "MAGI" #the name of the class that should be used to instantiate this application
            self.app_name = "MAGI"
            self.revision = 0 #the revision of the prompts and other metadata; increase this when the prompts are changed 

            self.system_prompt = system_prompt
            self.initial_prompt = initial_prompt
            self.assessment_template = assessment_template
            self.conversation_filter = ['assistant', 'user', c.MSG_SOURCE_QUERY, c.MSG_SOURCE_ANALYZE] #used to filter the conversation when building the prompt; only messages of these types will be included in the prompt
            self.tooling_descriptions = [] #will be created a bit later

            self.analyze_system_prompt = analyze_system_prompt
            self.library_prompt = library_prompt

            self.public = False #whether everyone can read, or only the owner
            self.owner = '' #only the authenticated owner can modify an assessment (but other owners can clone public assessments)


    def add_tooling_descriptions(self, tools):
        #automtically generate the tooling descriptions from the functions if they are not already in the template
        #this is useful when the agent is first created and the tooling descriptions are not yet in the template
        #schema['function']['name'] is the name of the function as it will be called by the LLM and must match the name of the function in the agent
        for f in tools:
            schema = get_openai_schema(f)
            self.tooling_descriptions.append(schema)
            logger.debug(json.dumps(schema, indent=4))  

    def save_to_db(self):
        """Save the application template to the mongo database"""
        template = save_template_todb(self.app_name, vars(self))
        self.revision = template['revision'] #update the revision number after saving the template
                           
    def load_from_db(self, name, revision=None):
        """Load the application template from the mongo database; use the latest version if version is not specified"""
        template  = get_template_fromdb(name, revision)
        for key in template:
            setattr(self, key, template[key])
        

class SessionState:  
    def __init__(self, app_name=None, revision=None):

        #gather all the metadata that will influence LLM output in one place
        self.a = ApplicationTemplate(app_name, revision) 

        self.llm = c.LLMModel() 
        #self.llm = c.LLMModel(provider=c.LLMProvider.HUGGINGFACE, model=c.HF_LLAMA_3_3_70B_INSTRUCT) 
        #self.llm = c.LLMModel(provider=c.LLMProvider.ANTHROPIC, model=c.A_CLAUDE_3_5_SONNET)

        self.STRUCTURED_OUTPUT = True 

        #create a unique id based on the current date and time; this will be used for persisting the state in the database
        #an empty string means that the state has not been saved yet
        self.name = ""  
        self.library = Library()
        self.library.library_prompt = self.a.library_prompt
        self.library.project_name = self.name

        self.public = False #whether everyone can read, or only the owner
        self.owner = '' #only the authenticated owner can modify an assessment (but other owners can clone public assessments)

        #prepare the tooling functions available to the agent
        tools = [self.Editor, self.Librarian, self.Critique] #list of tooling functions available to the LLM
        self.prepare_tools(tools)
        if not self.a.tooling_descriptions:  
            #the tooling description should be saved in the database as part of the app template and will be automatically loaded
            #if it is not there, we need to create it automatically from the code using pydantic
            self.a.add_tooling_descriptions(tools)

        #initialize the conversation
        self.current_turn = 0    #increases in respond() each time the user sends a message
        self.conversation = [] #record of the entire conversation, used for display in the chatbot GUI and to prepare the prompt for the next call to the LLM


        self.conversation.append({'role':'system', 'content':self.a.system_prompt, 'source':'assistant', 'tool_input':'', 'turn': self.current_turn, 'mm':self.a.assessment_template})
        self.conversation.append({'role':'assistant', 'content':self.a.initial_prompt, 'source':'assistant', 'tool_input':'', 'turn': self.current_turn, 'mm':self.a.assessment_template})
        
        #list of calls to the LLM, to be used for debugging in the Explain tab
        self.llm_call_list = []  
        
        self.mm = self.a.assessment_template #the mental map of the assistant is a blank copy of the Privacy Assessment template

        self.MAX_TURNS = 5 #maximum number of turns before returning control to the user

    def prepare_tools(self, tools):
        """Prepare the tools available to the agent; note: the names and parametes of the functions in the LLM schema must match the names of the functions in the agent, only the descriptions can change"""
        #prepare the tooling. We need to build
        #1) a list of tooling descriptions to pass to the LLM in the JSON forma expected by ChatGPT so that it knows what tools are available, their purpose and what inputs they expect
        #2) a dictionary of the functions to call for each tool {function_name:callable} where function_name is the name returned by the LLM
        self.tooling_functions = {}  
        for f in tools:
            self.tooling_functions[f.__name__] = f

    def format_conversation_message(self, output_msg):
        """format from AssistantAnswer to the conversation format"""
        msg = {'role':'assistant'}
        msg['source'] = output_msg.msg_source

        if self.STRUCTURED_OUTPUT:
            if output_msg.msg_source != c.MSG_SOURCE_ASSISTANT:
                msg['content'] = output_msg.msg_to_user 
            else:
                msg['content'] = f"Status:{str(output_msg.next_action)}\nAnswer:{output_msg.msg_to_user}"  #this is a message from the LLM, reflect the next_action and the message to the user
        else:
            msg['content'] = output_msg.msg_to_user #this is a tool call, make no assumption on the format of the message

        msg['tool_input'] = output_msg.tool_input
        msg['turn'] = self.current_turn
        msg['mm'] = self.mm   
        return msg

    def dumps(self):
        """return a JSON representation of the state for serialization"""
        d = {}
        
        d['app_name'] = self.a.app_name 

        d['name'] = self.name
        d['current_turn'] = self.current_turn #name+current_turn are used as a unique key in the database
        d['owner'] = self.owner
        d['public'] = self.public

        d['conversation'] = self.conversation
        d['llm_call_list'] = self.llm_call_list
        d['mm'] = self.mm

        d['llm'] = self.llm.__dict__
        d['llm'].pop('api_key', None)         #DO NOT  PERSIST THE LLM_CLIENT, the API key is secret.

        return d


    def log_llm_call(self, messages, tooling, chat_completion):
        """log the call to the LLM and the response received"""
        log={}
        log['messages']=messages
        log['tooling']=tooling
        log['chat_completion']=chat_completion.to_dict()  #does not work for HUGGINGFACE...?
        log['is_error']=False
        log['LlmProvider'] = self.llm.provider
        log['LlmModel'] = self.llm.model
        self.llm_call_list.append(log)

    def log_llm_call_error(self, messages, tooling, error):
        """log the call to the LLM and the response received"""
        log={}
        log['messages']=messages
        log['tooling']=tooling
        log['chat_completion']=error
        log['is_error']=True
        log['LlmProvider'] = self.llm.provider
        log['LlmModel'] = self.llm.model
        self.llm_call_list.append(log)

    def Editor(self,
        action:str = Field(description="Whether to replace the content of the identified element, delete it, or add a new sibbling element after it.", enum=["replace_content", "delete", "add_sibbling_after"]),
        id:str = Field(description="The id attribute of the existing element to replace, delete or add after."),
        content:str = Field(description="The content of the element to be replaced or added. When replacing the content of an element, the new content must be compatible with the element's tag; for example when replacing the content of a <tr> element, you should pass a string like '<td>...</td>' (but not the tag of the element itself). When adding a new element, include the tag of the new element with an id=new attribute.")
        ):
        """This tool allows the assistant to edit its notes (as an HTML document). It can replace the content of an identified element, delete it, or add a new sibbling element after it.
        When replacing the content of a complex element such as a table row, you must provide the entire row and include all relevant data fields in the replacement content, even if only one field has changed.        
        Only elements with an id attribute can be edited. The id's of elements are automatically computed by the function.
        """
        return editor(self, action=action, id=id, content=content) #note the closure on ss; this function will update ss.mm as a side-effect          

    def Librarian(self,
        query:str = Field(description="A question that this tool will answer based on available documents. The tool will perform a similarity search to find the most relevant information, and will return a summary."),
        category:str = Field(description="The category of documents to use for the query, empty string=all categories"),
        document_name:str = Field(description="The name of the only document to use for the query, empty string=all documents")
        ):
        """This tool answers a question by searching for relevant information in the library. It will perform a similarity search and summarize the excerpts returned by the search"""
        self.library.library_prompt = self.a.library_prompt

        #the LLM does not always put a value in there...
        if not isinstance(category, str):
            category = ""

        if not isinstance(document_name, str):
            document_name = ""

        answer = self.library.query(llm=self.llm, query=query, project_name=self.name, category=category, document_name=document_name, app_name=self.a.app_name) 
        return answer
    

    def Critique(self,
        focus:str = Field(description="A question or the section of the assessment to review")
    ):
        """This tool will perform a critical review of progress and will provide advice to meet the expectations of the user"""

        if not isinstance(focus, str):
            focus = ""

        messages = prompt(self, self.a.analyze_system_prompt, self.a.conversation_filter, prompt_method="CHAT")  
        focus = f"QUESTION OR SECTION TO REVIEW: {focus}"
        messages.append({'role':'user', 'content':focus})
       
        logger.info(f'Start Critique: {focus}')

        try:
            chat_completion = call_llm_once(self.llm, messages)
            explanation = chat_completion.choices[0].message.content
            self.log_llm_call(messages, [], chat_completion) 
            answer = c.AssistantAnswer(msg_source=c.MSG_SOURCE_ANALYZE, next_action=c.LLM_NEXT_ACTION_ASK_USER, msg_to_user=explanation)

        except  Exception as e:
            explanation = 'Error: ' + str(e)
            print(explanation, flush=True)
            self.log_llm_call_error(messages, [], explanation) 
            answer = c.AssistantAnswer(msg_source=c.MSG_SOURCE_ANALYZE, next_action=c.LLM_NEXT_ACTION_ASK_USER, msg_to_user=explanation)
        
        logger.info(f'End Critique: {focus}')

        yield answer

def rebase_id(soup):
    """Rebase the ids of all elements in the mental map document to ensure they are unique"""
    i = 1
    for el in soup.find_all(id=True):
        #logger.debug(print('mm:',el['id'], '->', el))
        el['id'] = str(i)
        i=i+1
    return soup

def mm_rebase_id(mm):
    soup = BeautifulSoup(mm,features="lxml")
    soup = rebase_id(soup)
    res = soup.prettify(formatter=None)  #to avoid escaping html tags
    return res

def editor(ss, action, id, content):  #side-effect: this function will update ss.mm
    """Edit a section of the agent's notes with the given action, id, and content.""" 
    status = c.AssistantAnswer(msg_source=c.MSG_SOURCE_EDITOR, tool_input=f"action={action}; id={id}; content={content}", next_action=c.LLM_NEXT_ACTION_PROCESS_INFORMATION, msg_to_user=f'OK: id={id} has been updated')
    soup = BeautifulSoup(ss.mm, features="lxml")
    el = soup.find(id=id)
    if el is None:
        status.msg_to_user = f'KO: element with id="{id}" not found'
    else:
        match action:
            case 'add_sibbling_after':
                #create a tag from the content string and append it after the element
                if content.startswith('<'):
                    new_tags = BeautifulSoup(content, features="lxml").body.contents #this parses the content as a fragment and returns a list of HTML tags
                    i=1
                    for tag in new_tags:

                        #get a unique id
                        found=True
                        i=1
                        while found:
                            new_id = f"{id}.{i}"
                            i+=1
                            if soup.find(id=new_id) is None:
                                found=False
                                break
                        tag['id'] = new_id  

                        el.insert_after(tag)    #potential problem? there could be further id added by the LLM in the children of the new tag...hopefully they will not conflict with ids in the original document
                        el = tag 

                else:
                    new_tag = soup.new_tag('div')
                    new_tag.string = content

                    found=True
                    i=1
                    while found:
                        new_id = f"{id}.{i}"
                        i+=1
                        if soup.find(id=new_id) is None:
                            found=False
                            break
                    new_tag['id'] = new_id  

                    el.insert_after(new_tag)
    
            case 'delete':
                el.decompose()

            case 'replace_content':
                el.string = content
        
            case _:
                status.msg_to_user = 'KO: unrecognized "action" value; valid inputs are "add_sibbling_after" | "delete" | "replace_content"'
   
    #do not rebase now, as the LLM may issue multiple commands in a single request, and all would use the original ids
    ss.mm = soup.prettify(formatter=None)  #formatter=None so that html tags that may appear in the content are not escaped
    yield status

def get_key(api_key_state, key_name):
    """Get an API key from the UI or the OS environment"""

    api_key = None
    if api_key_state is not None: #first try to get the key from the UI
        api_key = api_key_state.get(key_name, None)               
    if api_key is None: #then try to get it from the OS environment
        api_key = os.environ.get(key_name)
    return api_key

def check_api_key(ss, api_key_state):
    """Set the API key for the selected model"""

    match(ss.llm.provider):

        case c.LLMProvider.OPENAI:   
            ss.llm.api_key = get_key(api_key_state, "OPENAI_API_KEY")

        case c.LLMProvider.ANTHROPIC:
            ss.llm.api_key = get_key(api_key_state, "ANTHROPIC_API_KEY")

        case c.LLMProvider.HUGGINGFACE:
            ss.llm.api_key = get_key(api_key_state, "HUGGINGFACE_API_KEY")

        case c.LLMProvider.LAMBDALABS:
            ss.llm.api_key = get_key(api_key_state, "LAMBDALABS_API_KEY")

        case _:
            ss.llm.api_key = ''
    
    return ss.llm.api_key

def llm_loop(ss): #this generator will yield multiple updates that should be displayed in the user interface

    #prepare the prompt for the LLM; 
    #a filter allows you to select what categories of messages from the conversation will be included in the prompt
    #the prompt_method decides whether the prompt is a chat or a completion
    messages = prompt(ss, ss.a.system_prompt, ss.a.conversation_filter, prompt_method="CHAT") 
    messages = messages.copy() #make a copy to avoid polluting the original messages with the tool calls

    counter=0
    finished=False
    while not finished: #outer loop: the agent will continue to process the conversation until it decides to return control to the user; this may result in mutiple calls to the LLM

        logger.info(f"Agent enters iteration: {counter}; model: {ss.llm.provider}-{ss.llm.model}; structured output: {ss.STRUCTURED_OUTPUT}")


        #decide whether to use structured output to help the LLM decide whether to continue processing or to return to the user
        #this helps because the LLM does not always make tool calls but only returns an explanation of what it intends to do.
        response_format = c.LLMAnswer if ss.STRUCTURED_OUTPUT else None

        #disable tools before the last allowed turn to have the agent produce a message
        if counter >= ss.MAX_TURNS:
            logger.debug(f"Agent tools disabled after {counter} turns")
            td = None 
        else:
            td = ss.a.tooling_descriptions

        try:

            chat_completion = call_llm_once(ss.llm, messages, td, response_format=response_format) 
            ss.log_llm_call(messages, td, chat_completion) 
            logger.info(f"Completion: {chat_completion.__dict__}") #chat_completion.to_json()}") 

        except Exception as e:
            explanation = f'LLM failed to complete, with the following reason: {e}'
            ss.log_llm_call_error(messages, ss.a.tooling_descriptions, explanation)
            answer = c.AssistantAnswer(msg_source=c.MSG_SOURCE_ASSISTANT, next_action=c.LLM_NEXT_ACTION_ASK_USER, msg_to_user=explanation)
            logger.error(explanation)
            yield answer
            return

        #handle edge cases
        if chat_completion.choices[0].finish_reason not in ['stop', 'tool_calls']:
            explanation = f'LLM failed to complete, with the following reason: {chat_completion.choices[0].finish_reason}'
            logger.error(explanation)
            answer = c.AssistantAnswer(msg_source=c.MSG_SOURCE_ASSISTANT, next_action=c.LLM_NEXT_ACTION_ASK_USER, msg_to_user=explanation)
            ss.log_llm_call(messages, ss.a.tooling_descriptions, answer, after_tool_call=False) #after_tool_call=False means this is the first completion to the LLM, before tool calls
            yield answer
            return

        #now deal with the LLM's response
        response_message = chat_completion.choices[0].message
        tool_calls = response_message.tool_calls

        #return the LLM's message (if any) to the user
        #(note: we do not always get a message when the LLM makes tool calls, hence the need to check response_message.content)
        if response_message.content: 
            if ss.STRUCTURED_OUTPUT:
                msg = json.loads(response_message.content)
                next_action = c.LLM_NEXT_ACTION_PROCESS_INFORMATION if tool_calls else msg['next_action'] 
                msg_to_user = msg['msg_to_user'] #response_message.parsed.msg_to_user
            else:
                next_action = c.LLM_NEXT_ACTION_PROCESS_INFORMATION if tool_calls else c.LLM_NEXT_ACTION_ASK_USER 
                msg_to_user = response_message.content 
            yield c.AssistantAnswer(msg_source=c.MSG_SOURCE_ASSISTANT, next_action=next_action, msg_to_user=msg_to_user) #return to GUI the response from the LLM
        else:
            #there is no message yet from the LLM
            next_action = c.LLM_NEXT_ACTION_PROCESS_INFORMATION 

        #IMPORTANT: add the response message to the conversation in preparation of the next LLM call (if any)
        #this is expected by the LLM when there are tool calls.
        #however we do not want this in the UI, this is why the agent_response() function is maintaining a separate conversation list
        msg = response_message.to_dict() 
        if 'tool_calls' in msg and not msg['tool_calls']:
            msg.pop('tool_calls', None) #an empty tool_calls list causes OpenAI to crash
        messages.append(msg) 

        #now execute the tool calls and add their results to the conversation
        if tool_calls:  #some providers (eg Anthropic) may return None instead of an empty list when there are no tool calls
            for tool_call in tool_calls: 

                logger.debug(f"tool call by LLM: {tool_call}")

                function_name = tool_call.function.name
                function_to_call = ss.tooling_functions[function_name]  

                #test if arguments are a json string and convert them to a dictionary
                #needed for backward compatibility openai beta parse and openai chat completion
                function_args = tool_call.function.arguments
                if isinstance(function_args, str):
                    function_args = json.loads(function_args)

                #assume that a tool call call may yield multiple messages (maybe it is another agent...)
                #pass them up one by one but do not do anything with them...
                for function_response in function_to_call(**function_args):
                    logger.debug(f"tool call result: {function_response}")
                    yield function_response
        
                #...except with the last answer from the tool call, which we are going to pass to the LLM as the final result of this tool call
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool", 
                    "name": function_name,
                    "content": function_response.msg_to_user,
                }) 

            #now that all the tool calls have been executed, we need to clean up the ids of the mental map to ensure they are unique 
            ss.mm = mm_rebase_id(ss.mm)  
            messages = prompt_update_assistant_notes(ss, ss.mm, messages) #this ensures that the new version of the assistant note is correctly reflected in the prompt in case another call to the LLM is necessary

        #decide whether another call to the LLM is necessary and permitted
        if counter >= ss.MAX_TURNS: #return control to the user after a certain number of turns
            logger.debug(f"Agent loop forced to end")
            finished = True
        if next_action == c.LLM_NEXT_ACTION_ASK_USER: 
            finished = True
        #else: the LLM has made tool calls or decided to continue processing, so we need to call it again

        counter += 1

    logger.info(f"Agent loop ends: {counter}")

def agent_response(ss): 
    """This is the function called by the UI when the user submits a message. 
    The agent will process the user message and update its state (conversation and mental map) accordingly.
    This will yield a number of messages that need to be happened to the conversation and displayed in the UI, along with the edited mental map."""
    for output_msg in llm_loop(ss): 
        msg = ss.format_conversation_message(output_msg)
        ss.conversation.append(msg)
        yield ss, output_msg 



def get_library_index(ss):
    """Get an index of available documents from the library for display"""
    docs = ss.library.get_documents(project_name=ss.name, app_name=ss.a.app_name)
    res = "<table><tr><th>Document Name</th><th>Category</th><th>Abstract</th></tr>"
    for doc in docs:
        res += f"<tr><td>{doc['document_name']}</td><td>{doc['category']}</td><td>{doc['abstract']}</td></tr>"
    res += "</table>"
    return res

def prompt_update_assistant_notes(ss, mm, messages):
        #update the assistant note part of the prompt with the latest notes; this is called by call_llm_tool after the tool calls have been processed
        p = "ASSISTANT'S NOTES:```" + mm + "```\n"
        msg_index = 2 if ss.llm.provider == c.LLMProvider.ANTHROPIC else 1
        messages[msg_index]['content']=p #the assistant note is the second message in the list (see prompt() function)
        return messages

def prompt(ss, system_prompt, conversation_filter=None, prompt_method="CHAT"):
    messages = []
    messages.append({'role':'system', 'content':system_prompt})

    if ss.llm.provider == c.LLMProvider.ANTHROPIC: 
        #Anthropic requires the first message to be from the user...
        messages.append({'role':'user', 'content':'hello'})

    match prompt_method:
        case "CHAT":
            #use the conversation structure of chatgpt 
            p = "ASSISTANT'S NOTES:```" + ss.mm + "```\n"
            messages.append({'role':'system', 'content':p})
            p = "DOCUMENT LIBRARY:```" + get_library_index(ss) + "```\n"
            messages.append({'role':'system', 'content':p})
            for turn in ss.conversation:
                if turn['role'] != 'system' and (conversation_filter is None or turn['source'] in conversation_filter):
                    #if turn['role'] == 'assistant' and turn['source'] == 'assistant_thinking' and turn['turn'] != current_turn: 
                    #    continue #do not show the thinking messages from the previous turn
                    messages.append({'role':turn['role'], 'content':turn['content']})
            #p = "Let's first review the notes I have taken so far and the list of available documents:\n\n"

        case "FULL":
            #put everything, including the conversation into a single message
            p = "ASSISTANT'S NOTES:```" + ss.mm + "```\n"
            p += "DOCUMENT LIBRARY:```" + get_library_index(ss) + "```\n"
            p += "CONVERSATION:```\n"
            for turn in ss.conversation:
                if turn['role'] != 'system' and (conversation_filter is None or turn['source'] in conversation_filter):
                    #if turn['role'] == 'assistant' and turn['status'] == 'assistant_thinking' and turn['turn'] != current_turn: 
                    #    continue #do not show the thinking messages from the previous turn
                    p += f"{turn['role']}: {turn['content']}\n"
            p += '```\n'
            messages.append({'role':'system', 'content':p})

    return messages


def explain(ss, call_details, question):

    #prepare the prompt to obtain an explanation
    p=[]
    messages = call_details['messages']
    completion = call_details['chat_completion']

    p.append({'role':'system', 'content':"You are an expert in LLM prompt engineering. You help the user understand the chat completion from a call to a LLM"})

    s="DETAILS OF THE LLM CALL: Here is some information about the tools that were available to the LLM, the conversation used to prompt the LLM, and the resulting chat completion\n"
    
    s += 'Tool Descriptions:```\n'
    if call_details['tooling']:
        for t in call_details['tooling']:
            s += str(t) + "\n"
        s += '```\n'    

    s +="Prompt:```\n"
    if messages:
        for m in messages:
            s += str(m) + "\n---\n"  #messages can be ChatCompletionObjects or tool call dictionaries
        s += '```\n'    

    s+="Chat Completion:```\n"
    #data = completion.model_dump_json(indent=3) #if logging a ChatCompletion object
    if call_details['is_error']:
        data = completion
    else:
        data = json.dumps(completion,indent=3) #if logging a dict
    s += data
    p.append({'role':'assistant', 'content':s})
    s += '```\n'    
    
    p.append({'role':'user', 'content':question})

    try:
        #chat_completion = ss.llm_client.chat.completions.create(
        #    messages=p,
        #    model=ss.llm.model,
        #    temperature=c.TEMPERATURE,
        #    response_format=None
        #)
        chat_completion = call_llm_once(ss.llm, messages=p)
        explanation = chat_completion.choices[0].message.content
    
    except  Exception as e:
        logger.error('Error in explain(): ' + str(e))
        #print('Error: ' + str(e), flush=True)
        return 'Error: ' + str(e)
    
    return explanation




def create_app_template():

    app = SessionState()

    app.a.app_name = "MAGI"
    app.public=True
    app.owner='Arnadu'

    app.a.public=True
    app.a.owner='Arnadu'
    app.a.save_to_db()

    return app

if __name__ == "__main__":

    #uncomment this to save a template app to the database
    #create_app_template()

   
    #uncomment this to instantiate a test app
    #s = SessionState(app_name="Demo Assistant")
    pass
