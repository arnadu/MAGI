from datetime import datetime
import os
import json

from bs4 import BeautifulSoup

import gradio as gr

#import openai
#from huggingface_hub import InferenceClient

import constants as c
from  agent import agent_response, explain, rebase_id
from  agent import get_library_index, get_list_of_assessments, save_assessment_todb, load_assessment_fromdb
from agent import check_api_key
from database import get_list_of_app_templates, get_revisions_of_app_template, get_template_fromdb
#from database import save_template_todb

import logging
import sys

from key_store import head_js, decrypt_value

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.debug("Starting the ui module")


#there are three main data structures

#the LLM itself returns a JSON object with the following structure defined in constants.LLMAnswer
# next_action: "process_information|ask_user",
# msg_to_user: "the message to the user",

#the functions agent_response(), call_llm_tool() and also tools return a dictionary with the following structure defined in constants.AssistantAnswer
# msg_source: str = MSG_SOURCE_ASSISTANT  | tool name (see constants)
# tool_input: str = the input to the tool when the message is a tool call
# next_action: str = process_information | ask_user (see constants)
# msg_to_user: str = the message to the user or the result of the tool call; this can be (Status:next_action\nAnswer:msg_to_user) if from assistant, or just msg_to_user if from a tool call



#Messages in the SessionState.conversation list can be a user message or an assistant message
#the format of these messages is:
# role : system|assistant|user, 
# source: assistant|EditorTool|QueryDocuments|..., 
# content: the output of the tool call, or the formatted LLM result eg next_action:...\nmsg_to_user:..., 
# tool_input: the inputs to a tool call
# turn: the turn number
# mm: the resulting mental map 




def format_chat_message(msg):
    """Format a single chat message for display in the chatbot"""
    if msg['role'] == 'user':
        return {'role': 'user', 'content': f"({msg['turn']}) {msg['content']}"} #preprend the turn number to the user message so that the user can easily refer to it in the rollback dropdown
    elif msg['role'] == 'assistant':
        if msg['source'] != c.MSG_SOURCE_ASSISTANT: #decorate tool calls with metadata
            desc = f"Tool: {msg['source']}\n"
            desc += f"Input: {msg['tool_input']}\n"
            desc += f"Answer: {msg['content']}"
            return {'role': 'assistant', 'content': desc, 'metadata': {'title': f"🛠️ {msg['source']}"}}
        else: #regular assistant message
            return {'role': 'assistant', 'content': msg['content']}

def conversation_for_display(conversation):
    """Prepare the entire conversation for display in the chatbot"""
    res = []
    for turn in conversation:
        if turn['role'] in ['assistant','user']:
            res.append(format_chat_message(turn))
    return res

def mm_for_display(mm, rebase=False):
    """Prepare the mental map for display in the HTML viewer"""
    soup = BeautifulSoup(mm,features="lxml")
    if rebase:
        soup = rebase_id(soup)
    res = soup.prettify(formatter=None)
    return res



css = '''

.gradio-container {background-color: white !important;}

.column-form .wrap {flex-direction: column;} /*for sidebar; see https://huggingface.co/spaces/Gradio-Community/sidebar_with_gradio2/blob/main/app.py*/

#doc_viewer {background-color: white !important;}

.tool-button { /*use this to show icons*/
    background-color: white !important; /*force a white-background so that only the icon is visible*/
    color: black !important;
    border-color: black !important;
    max-width: 50pt !important;
    min-width: 50pt !important;
    /*min-width: 2.2em !important;*/
    /*height: 2.4em !important;*/
    /*align-self: end;*/
    /*line-height: 1em;*/
    border-radius: 1em;
}
'''

def make_model_list():
    """make a list of availabel provide:model pairs"""
    models = []
    models.append(f"{c.LLMProvider.OPENAI}:{c.OAI_GPT_4o}")
    models.append(f"{c.LLMProvider.OPENAI}:{c.OAI_GPT_o1}")
    models.append(f"{c.LLMProvider.ANTHROPIC}:{c.A_CLAUDE_3_5_SONNET}")
    models.append(f"{c.LLMProvider.ANTHROPIC}:{c.A_CLAUDE_3_OPUS}")
    models.append(f"{c.LLMProvider.HUGGINGFACE}:{c.HF_LLAMA_3_3_70B_INSTRUCT}")
    return models


#note: the same logic is also implemented in js
def parse_provider_model(provider_model):
    """parse a provider:model string into a tuple"""
    provider, model = provider_model.split(":")
    return provider, model



def ui(make_session_state, app_name, revision=None):
    #theme=gr.themes.Origin()
    #theme='argilla/argilla-theme'
    #theme='theNeofr/Syne'
    
    logger.debug(f"Creating UI for app: {app_name}, revision: {revision}")

    gr.set_static_paths(paths=["assets/"])    

    with gr.Blocks(css=css, head=head_js) as demo:  

        ss = gr.State(value=make_session_state(app_name, revision))  #start the UI with a blank session state of the specified app template
        user_info = gr.State(value={"username":None})  #keet track of authenticated user

        with gr.Row():

            with gr.Column(scale=3) as sidebar:  #sliding sidebar

                gr.Button(value="User Guide", link="https://github.com/arnadu/MAGI/blob/main/DOC.md")

                with gr.Accordion(label='Login', open=True):

                    gr.Markdown("Warning - logging-in will restart your session without saving")
                    login_btn = gr.LoginButton()

                with gr.Accordion(label='LLM Model', open=True):

                    gr.Markdown("Select your model provider and enter your API key (it is not saved to the database).")
                    llm_provider = gr.Dropdown(show_label=False, choices=make_model_list(), container=False)
                    api_key = gr.Textbox(show_label=False, type="password", container=False)
                    str_out = gr.Checkbox(ss.value.STRUCTURED_OUTPUT, label="Structured Ouput", container=False)

                    @str_out.change(inputs=[ss, str_out], outputs=[ss])
                    def str_out_change(ss, str_out):
                        ss.STRUCTURED_OUTPUT = str_out
                        return ss

                    def set_model_key(ss, provider_model, encrypted_key):
                        #update the session state with the selected LLM provider and model
                        #then decrypt the key and store it in the session state
                        provider, model = parse_provider_model(provider_model)
                        ss.llm.provider = provider
                        ss.llm.model = model
                        if encrypted_key:
                            key = decrypt_value(encrypted_key)
                            ss.llm.api_key = key
                        else:
                            ss.llm.api_key = None
                        return ss

                    #retrieve the (encrypted) API key for the selected LLM provider from the browser's local storage if it exists already and pass it to the gradio session
                    #for a funky reason, .then does not trigger if there is only js in the preceding event, hence the need for a dummy lambda, which in turn requires fooling around with the params
                    llm_provider.change(fn=lambda _,y: y, inputs=[llm_provider, api_key], outputs=[api_key],
                        js="""(llm_provider, dummy) => {
                            //tretrieve this LLM provider (encrypted) API key from the browser's local storage if it exists already
                            const key_name = llm_provider.split(':')[0]; 
                            const key_value = localStorage.getItem(key_name); 
                            console.log('retrieved ', key_name, key_value); 
                            return [llm_provider, key_value || '']; 
                        }"""
                    ).then(fn=set_model_key, inputs=[ss, llm_provider, api_key], outputs=[ss])

                    #acquire the API key from the user (the .submit event is triggered on pressing <enter>), encrypt it, store it in the browser's local storage and pass it to the gradio backend 
                    api_key.submit(fn=lambda _,y:y, 
                                   inputs=[llm_provider, api_key], 
                                   outputs=[api_key],
                                    js="""async (llm_provider, api_key) => {
                                        //encrypt the API key with the public key and store it in the browser's local storage
                                        //return the encrypted value to the UI, this is what the backend will get
                                        //the backend will decrypt it using the private key
                                        const key_name = llm_provider.split(':')[0];
                                        const encrypted = await encryptValue(api_key);
                                        console.log('encrypted ', api_key, ' --> ', encrypted); 
                                        console.log('saving ', key_name, encrypted); 
                                        localStorage.setItem(key_name, encrypted); 
                                        return [llm_provider, encrypted];
                                    }"""
                    ).then(fn=set_model_key, inputs=[ss, llm_provider, api_key], outputs=[ss])

                with gr.Accordion(label='Assessment', open=True):

                    gr.Markdown("Create a new assessment (select the application template)")
                    app_dd = gr.Dropdown(get_list_of_app_templates(), show_label=False, container=False)
                    create_assessment_btn = gr.Button("New", size="sm")

                    gr.Markdown("Or load an existing assessment")
                    load_assessment_dd = gr.Dropdown(get_list_of_assessments(), show_label=False, container=False)
                    load_assessment_btn = gr.Button("Load", size="sm")
                    
                    gr.Markdown("Save your assessment (any existing assessment of the same name will be overwritten)")
                    assessment_name = gr.Textbox(show_label=False, container=False)
                    public_assessment_cb = gr.Checkbox(value=False, label="public", container=False)
                    save_as_btn = gr.Button("Save", size="sm")


                with gr.Accordion(label='Documents', open=True):

                    gr.Markdown("Upload a document into the library; select the appropriate category and provide an abstract of the document.\nYour assessment needs to be saved before you can upload documents to the library.")

                    file_picker = gr.File(label="Upload a file", file_count="single", container=False)
                    document_name = gr.Textbox(label="Document Name", container=False)
                    document_category = gr.Dropdown(["Law", "Policy", "Notice", "Contract", "Assessment", "Other"], label="Category", container=False)
                    document_abstract = gr.TextArea(label="Abstract", lines=3, container=False)

                    file_picker.change(
                        lambda path : "" if path is None else os.path.basename(path),
                        inputs=[file_picker], outputs=[document_name]
                    )

                    app_default_cb = gr.Checkbox(label="App default", value=False, container=False)

                    upload_button = gr.Button("Upload", size="sm", interactive=False)

                    document_list = gr.HTML(get_library_index(ss.value))


            with gr.Column(scale=9) as main_panel:

                with gr.Row(): 
                    open_sidebar_btn = gr.Button(">>", elem_classes='tool-button', visible=False,scale=0)
                    close_sidebar_btn = gr.Button("<<", elem_classes='tool-button', visible=True, scale=0)

                    title = gr.Markdown(f"# {app_name}: ")   

                    open_sidebar_btn.click(lambda: {
                            open_sidebar_btn: gr.Button(visible=False),
                            close_sidebar_btn: gr.Button(visible=True),
                            sidebar: gr.Column(visible=True)
                        }, 
                        outputs=[open_sidebar_btn, close_sidebar_btn, sidebar]
                    )
                        
                    close_sidebar_btn.click(lambda: {
                            open_sidebar_btn: gr.Button(visible=True),
                            close_sidebar_btn: gr.Button(visible=False),
                            sidebar: gr.Column(visible=False)
                        }, 
                        outputs=[open_sidebar_btn, close_sidebar_btn, sidebar]
                    )

                with gr.Tab("Chat"):
                    with gr.Row():
                        with gr.Column(): #conversation
                            with gr.Group():
                                avatar_images=["assets/user_avatar.png", "assets/system_avatar.png"]
                                chatbot = gr.Chatbot(type="messages", show_label=False, value=conversation_for_display(ss.value.conversation), avatar_images=avatar_images)
                                with gr.Row(equal_height=True):
                                    msg = gr.MultimodalTextbox(placeholder="Type your message here", submit_btn=True, show_label=False, scale=12)
                                    stop_btn = gr.Button(value='', icon='assets/stop.svg', size='sm', scale=0)
                                    
                            with gr.Group():
                                gr.Markdown("You can reset the conversation to any prior state by selecting the corresponding turn number. Documents uploaded to the library will not be deleted.")
                                roll_back_dd = gr.Dropdown([], label="Roll Back")

                        with gr.Column(): #mental map
                            gr.Markdown('Edit the ASSISTANT NOTES to add or correct what the AI wrote. You can also provide structure to the document by adding id="1" tags in your elements: this will enable to the AI to change them. Use elements with class="Instructions" to provide detailed instructions to the AI.')
                            edit_toggle_btn = gr.Button("Edit", size='sm')
                            mm_editor = gr.Code(label=None, value=ss.value.mm, language="html", visible=False)
                            doc_viewer = gr.HTML(mm_for_display(ss.value.mm, rebase=True), elem_id="doc_viewer", visible=True)

                            #@rebase_btn.click(inputs=[ss, mm_editor], outputs=[ss, mm_editor, doc_viewer])
                            #def rebase_mm(ss, content):
                            #    ss.mm = content
                            #    html = mm_for_display(ss.mm, rebase=True)
                            #    editor = gr.Code(label="Mental Map Editor", value=html, language="html")
                            #    return ss, editor, html

                            @edit_toggle_btn.click(inputs=[ss, mm_editor, edit_toggle_btn], outputs=[ss, mm_editor, doc_viewer, edit_toggle_btn])
                            def edit_toggle(ss, mm_editor, edit_toggle_btn):
                                if edit_toggle_btn == 'View': #we are in Edit mode
                                    #toggle to View mode
                                    ss.mm = mm_for_display(mm_editor, rebase=True)  #capture the changes made by the user
                                    editor = gr.Code(value=ss.mm, visible=False)
                                    viewer = gr.HTML(value=ss.mm, visible=True)
                                    btn = gr.Button(value="Edit")
                                else: #we are in View mode
                                    #toggle to Edit mode
                                    editor = gr.Code(value=ss.mm, visible=True) #in case it has been changed by the assistant
                                    viewer = gr.HTML(visible=False)
                                    btn = gr.Button(value="View")
                                return ss, editor, viewer, btn

                with gr.Tab("App Designer"):

                    gr.Markdown("Load an existing Application template")
                    with gr.Row(equal_height=True):
                        template_name_dd = gr.Dropdown(get_list_of_app_templates(), label="Application")
                        template_revision_dd = gr.Dropdown([], label="Revision")
                        load_template_btn = gr.Button("Load", size="sm", scale=0)

                    gr.Markdown("Save this Application template; select the Application Class that can execute the template")
                    with gr.Row(equal_height=True):
                        new_template_name = gr.Textbox(label="Application Name", placeholder="Application Name")
                        public_template_cb = gr.Checkbox(value=False, label="is public")
                        save_template_btn = gr.Button("Save", size="sm", scale=0)

                    with gr.Group():    
                        with gr.Group():
                            gr.Markdown("Adapt the overall system prompt of the LLM, as well as the initial prompt to get the conversation started with the user")
                            system_prompt = gr.TextArea(label="System Prompt", lines=10)
                            initial_prompt = gr.TextArea(label="Initial Prompt", lines=1)

                        with gr.Group():
                            gr.Markdown("The assistant mental map; this is where you should structure the outline of the assessment with detailed Instructions")
                            assessment_template = gr.Code(label="Assessment Template", language="html")

                        with gr.Group():
                            gr.Markdown("The system prompt used for the Librarian tool to answer a question using the documents in the library")
                            library_prompt = gr.TextArea(label="Librarian prompt")

                        with gr.Group():
                            gr.Markdown("Describe to the LLM how to use the Critique Tool and its <Focus> parameter")
                            analyze_description = gr.TextArea(label="Tool Description", lines=3)
                            analyze_param = gr.TextArea(label="Focus Parameter", lines=2)
                            analyze_system_prompt = gr.TextArea(label="LLM System Prompt", lines=10)

                    @template_name_dd.change(inputs=[template_name_dd], outputs=[template_revision_dd])
                    def update_template_revision_dd(template_name):
                        return gr.Dropdown(get_revisions_of_app_template(template_name), label="Revision")

                    @load_template_btn.click(inputs=[template_name_dd, template_revision_dd], 
                                            outputs=[new_template_name, public_template_cb, system_prompt, initial_prompt, assessment_template, analyze_description, analyze_param, analyze_system_prompt, library_prompt])
                    def load_template(template_name, revision):
                        template = get_template_fromdb(template_name, revision)
                        analyze_desc = ""
                        analyze_param = ""
                        analyze_system_prompt = ""
                        for t in template['tooling_descriptions']:
                            f = t['function']
                            if f['name'] == "Critique":
                                analyze_desc = f['description']
                                p = f['parameters']['properties']
                                analyze_param = p['focus']['description']
                                analyze_system_prompt = template['analyze_system_prompt']
                                break   
                        
                        return (
                            template['app_name'],
                            template['public'],
                            template['system_prompt'],
                            template['initial_prompt'],
                            template['assessment_template'],
                            analyze_desc,
                            analyze_param,
                            analyze_system_prompt,
                            template['library_prompt'], 
                        )

                    @save_template_btn.click(inputs=[user_info, new_template_name, system_prompt, initial_prompt, assessment_template, analyze_description, analyze_param, analyze_system_prompt, library_prompt, public_template_cb], 
                                            outputs=[template_name_dd, template_revision_dd, app_dd])
                    def save_template(user_info, name, sys_prompt, init_prompt, assessment_template, analyze_description, analyze_param, analyze_system_prompt, library_prompt, is_public):    
                        
                        if not name:
                            raise gr.Error("Please enter an Application name")
                        
                        if not user_info or not user_info["username"]:
                            raise gr.Error("Please login")
                        
                        # Create a new instance of the application class
                        ss = make_session_state()
                        
                        # Set the template fields
                        ss.a.app_name = name
                        ss.a.system_prompt = sys_prompt
                        ss.a.initial_prompt = init_prompt
                        ss.a.assessment_template = assessment_template
                        ss.a.public = is_public
                        ss.a.owner = user_info["username"]
                        
                        ss.a.analyze_system_prompt = analyze_system_prompt
                        for t in ss.a.tooling_descriptions:
                            f = t['function']
                            if f['name'] == "Critique":
                                f['description'] = analyze_description
                                p = f['parameters']['properties']
                                p['focus']['description'] = analyze_param
                                break   

                        ss.a.library_prompt = library_prompt

                        # Save to database
                        ss.a.save_to_db()
                        
                        # Update dropdowns

                        return (
                            gr.Dropdown(get_list_of_app_templates(user_info["username"]), value=name),
                            gr.Dropdown(get_revisions_of_app_template(name), value="Latest"),
                            gr.Dropdown(get_list_of_app_templates(user_info["username"])),
                        )

                with gr.Tab("Explain"):
                    with gr.Group():    
                        gr.Markdown("This tab allows you to get an explanation of the LLM call that led to a specific chat completion. Select the LLM call number you want to debug")
                        llm_call_number = gr.Number(label="LLM Call Number", value=0, precision=0)
                        messages = gr.Textbox(label="Messages", value="Messages", interactive=False)
                        completion = gr.Textbox(label="Completion", value="Completion", interactive=False)
                        prompt_text = gr.Textbox(label="Prompt", value="Prompt", interactive=False)

                    gr.Markdown("Ask the LLM to explain itself and to suggest modifications to the prompt to avoid whatever it is you do not like about its answer")
                    question = gr.TextArea(label="Your Question", lines=10, value="Why did the assistant reply XXX? I would have expected YYY instead. Identify the specific part(s) of the prompt or instructions that triggered the unexpected response. Then suggest a more robust wording.")
                    explain_btn = gr.Button("Explain", size='sm', scale=0)
                    answer = gr.HTML("Answer")

                    def retrieve_llm_call(ss, llm_call_number):
                        i = int(llm_call_number)
                        l = len(ss.llm_call_list)
                        
                        #show the response only
                        message = ""
                        completion = ""
                        prompt=""
                        if l>0 and i>=0 and i<l:
                            call_details = ss.llm_call_list[i]
                            if call_details['is_error']:
                                completion = call_details['chat_completion']
                            else:
                                completion = json.dumps(call_details['chat_completion'], indent=3) 
                                message = call_details['chat_completion']["choices"][0]["message"]["content"] 
                            for m in call_details['messages']:
                                prompt += str(m) + "\n---\n"  #messages can be ChatCompletionObjects or tool call dictionaries
                        else:
                            message = "No such LLM call number"
                            completion = ""

                        return ss, message, completion, prompt, "" #return info about the selected call, clear the explanation answer
                    
                    llm_call_number.change(retrieve_llm_call, inputs=[ss, llm_call_number], outputs=[ss, messages, completion, prompt_text, answer])
                    
                    @explain_btn.click(inputs=[ss, llm_call_number, question], outputs=[ss, answer])
                    def explain_llm_call(ss, llm_call_number, question):
                        check_api_key(ss)
                        i = int(llm_call_number)
                        l = len(ss.llm_call_list)
                        explanation = "No such LLM call number"
                        if l>0 and i>=0 and i<l:
                            call_details = ss.llm_call_list[i]
                            explanation = explain(ss, call_details=call_details, question=question)
                        return ss, explanation


        @upload_button.click(inputs=[ss, user_info, file_picker, document_name, document_category, document_abstract, app_default_cb], 
                            outputs=[ss, file_picker, document_name, document_category, document_abstract, document_list])
        def upload_document(ss, user_info, temp_file_path, document_name, document_category, document_abstract, app_default_cb):
            check_api_key(ss)
            """Upload a document into the library."""
            if app_default_cb:
                project_name = f"{ss.a.app_name}.*" #make the document avaiable by default to all assessments for this application
            else:
                project_name = ss.name
            print(f"uploading document: {temp_file_path} to app/project: {project_name}")
            ss.library.upload_file(project_name=project_name, document_name=document_name, category=document_category, filepath=temp_file_path, abstract=document_abstract)    

            return ss, None, None, None, None, get_library_index(ss)


        @save_as_btn.click(inputs=[ss, user_info, assessment_name, public_assessment_cb], outputs=[ss, load_assessment_dd, title, public_assessment_cb])
        def save_assessment_as(ss, user_info, new_name, is_public):
            old_name = ss.name
            #TODO: figure out how to get gradio to popup a modal confirmation dialog?
            #if ss.library.project_exists(new_name):
            #    if not gr.Warning(f"Project '{new_name}' already exists. Do you want to overwrite it?").then():
            #        return ss, gr.Textbox(label="Assessment Name", value=old_name, interactive=True), gr.Dropdown(get_list_of_assessments(), label="Assessment List")
            ss.name = new_name
            ss.library.project_name = new_name
            if old_name != new_name:
                ss.library.clone_project(old_name, new_name)
            ss.public = is_public
            ss.owner = user_info["username"]
            data = ss.dumps()
            res = save_assessment_todb(ss.name,data)
            dd = gr.Dropdown(get_list_of_assessments(), show_label=False, scale=1)
            cc = gr.Checkbox (ss.public)
            title = f"# {ss.a.app_name}: {ss.name}"
            return ss, dd, title, cc



        @roll_back_dd.change(inputs=[ss, roll_back_dd], outputs=[ss, chatbot, msg, mm_editor, doc_viewer, roll_back_dd])
        def roll_back(ss, roll_back):
            #find the user message and mm at the selected turn (there may be several messages for a given turn in the conversation, we pick the first one)
            turn_start = [turn for turn in ss.conversation if turn['role']=='user' and turn['turn']==roll_back][0]

            #roll back the conversation to the selected turn
            conversation = [msg for msg in ss.conversation if msg['turn']<roll_back]

            #roll back the conversation to the selected turn
            ss.mm = turn_start['mm'] #the mental map is rolled back to the state at the selected turn
            html = mm_for_display(ss.mm, rebase=True)
            editor = gr.Code(label="Mental Map Editor", value=html, language="html")
            ss.current_turn = roll_back-1
            ss.conversation = conversation
            msg = turn_start['content'] #retrieve the user message that was sent to the llm to initiate this turn
            dd = gr.Dropdown(range(1, roll_back+1), label="Roll Back")

            return ss, conversation_for_display(ss.conversation), msg, editor, html, dd

        @create_assessment_btn.click(inputs=[ss, app_dd], outputs=[ss, chatbot, msg, mm_editor, doc_viewer, document_list, assessment_name, roll_back_dd, title])
        def create_assessment(ss, app_name):
            ss = make_session_state(app_name) #reset the session state
            
            #update the UI
            html = mm_for_display(ss.mm, rebase=True)
            editor = gr.Code(label="Mental Map Editor", value=html, language="html")
            dd = gr.Dropdown(range(1, ss.current_turn+1), label="Roll Back")

            title=f"# {ss.a.app_name}: {ss.name}"
            
            return ss, conversation_for_display(ss.conversation), "", editor, html, get_library_index(ss), gr.Textbox(value=ss.a.app_name), dd, title
        

        @load_assessment_btn.click(inputs=[ss, load_assessment_dd], outputs=[ss, chatbot, msg, mm_editor, doc_viewer, document_list, assessment_name, roll_back_dd, title])
        def load_assessment(ss, name):
            ss = make_session_state(app_name) #reset the session state; this loads the app template too: get the latest one by default
            assessment, template = load_assessment_fromdb(name)
            
            #load the session data into the session
            #TODO: this could be moved to agent.py, getting very intimate with the internals!
            ss.__dict__.update(assessment) #update the session state with the loaded data
            ss.llm = c.LLMModel(**ss.llm)
            ss.library.project_name = name

            #update the UI
            html = mm_for_display(ss.mm, rebase=True)
            editor = gr.Code(label="Mental Map Editor", value=html, language="html")
            dd = gr.Dropdown(range(1, ss.current_turn+1), label="Roll Back")
            title = f"# {ss.a.app_name}: {ss.name}"
            return ss, conversation_for_display(ss.conversation), "", editor, html, get_library_index(ss), gr.Textbox(value=name), dd, title


        def respond(ss, msg, editor): 
            """Handler of the "submit" button in the user interface. 
            This will make the agent process the user input and handle the refresh of the UI as answer messages are coming through"""

            #make sure there is a LLM available
            check_api_key(ss)

            #direct user message to the LLM
            content = msg['text']

            #if the user has uploaded files in the chat, add them to the user message
            files = msg['files']
            if len(files) > 0:
                import pymupdf4llm
                content += "\n\nFILE(S) UPLOADED BY USER"
                for filepath in files:
                    content += 'file:' + filepath + "\n" + '```'
                    md_text = pymupdf4llm.to_markdown(filepath)  # get markdown for all pages
                    content += md_text + '```\n'

            msg = content


            #capture any edit made by the user to the mental map
            html = mm_for_display(ss.mm, rebase=True) 
            ss.mm = html

            #add the user's prompt to the conversation adn create a snapshot of current state so that we can roll back to this point
            ss.current_turn += 1 

            ss.conversation.append({'role':'user', 'source':'user', 'content':msg, 'tool_input':'', 'turn':ss.current_turn, 'mm':ss.mm}) 

            #clean the ids of the mental map to make sure they are unique, return the mm in html format

            #first update to the user interface; this simply moves the user message from the input box to the chat history in the user interface
            chat_history = conversation_for_display(ss.conversation) #prepare the conversation for display in the chatbot
            editor = gr.Code(label="Mental Map Editor", value=ss.mm, language="html")
            dd = gr.Dropdown(range(1, ss.current_turn+1), label="Roll Back") #rollback dropdown needs to include the current turn
            msg = "" #clear the input box (the user message has been moved to the chat history)
            yield ss, msg, chat_history, editor, html, dd

            #now handle all updates from the LLM as it processes the user message
            for ss, output_msg in agent_response(ss):
                
                #format the output message and add it to the chat history in the user interface; use metadata to decorate tool calls
                conv_msg = ss.format_conversation_message(output_msg)
                chat_msg = format_chat_message(conv_msg)
                chat_history.append(chat_msg)

                #update the viewers of the mental map
                html = mm_for_display(ss.mm, rebase=False) #ss.mm may have been updated by the tooling function; do not change the ids of the nodes to faciliate debugging (and we should not touch ss.mm itself from here!)
                editor = gr.Code(label="Mental Map Editor", value=html, language="html")

                yield ss, msg, chat_history, editor, html, dd  

            #finalize the run by updating the UI with the final state of the conversation
            html = mm_for_display(ss.mm, rebase=True) #rebase the ids of the nodes in the mental map
            editor = gr.Code(label="Mental Map Editor", value=html, language="html")
            yield ss, msg, chat_history, editor, html, dd  

        #enables the user to interrupt the agent's processing of the user input
        run_event = msg.submit(fn=respond, inputs=[ss, msg, mm_editor], outputs=[ss, msg, chatbot, mm_editor, doc_viewer, roll_back_dd])
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event]) 

        def login(user_info, profile: gr.OAuthProfile | None):
            #we use gradio's OAuth authentication of HF Spaces to authenticate users https://www.gradio.app/guides/sharing-your-app#authentication

            #we need a user to be authenticated in order to allow saving to the database. The following interface elements must be inactive if the user is not known
            #saving an assessment (save_as_btn)
            #saving a file to the library (file_picker and upload_btn)
            #saving a template (save_template_btn)

            #furthermore, several dropdowns should show different lists depending on the user
            #available assessments (load_assessment_dd): all public assessments + the user's private ones
            #availale application templates to create a new assessment (app_dd): all public templates + the user's private ones
            #availale application templates to load in the app designer (app_template_dd): all public templates + the users' privacy ones

            #additional considerations
            #only the owner of an app template should be able to add documents to this app's library

            logger.info(f"Logging with profile: {profile}")

            username = profile.username if profile else None
            user_info['username'] = username

            app_templates_list = get_list_of_app_templates(username)

            app_dd = gr.Dropdown(choices=app_templates_list)
            load_assessment_dd = gr.Dropdown(choices=get_list_of_assessments(username))
            template_name_dd = gr.Dropdown(choices=app_templates_list)

            if not username:
                #deactivate the UI elements that require to be authenticated
                return user_info, gr.Button(interactive=False), gr.File(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False), app_dd, load_assessment_dd, template_name_dd
            else:
                #activate UI elements that require the user to be known
                return user_info, gr.Button(interactive=True), gr.File(interactive=True), gr.Button(interactive=True), gr.Button(interactive=True), app_dd, load_assessment_dd, template_name_dd

        #.then() does not seem to trigger unless there is a valid python function, even when there is a js function
        #hence the need for a dummary lambda x,y:y function
        #there is also a subtlety about why two params are needed; lambda x:x, inputs=[llm_provider], outputs=[api_key], js=(llm_provider) => { return api_key } does not work. the api_key ends up in llm_provider...
        demo.load(fn=login, 
                    inputs=[user_info], 
                    outputs=[user_info, save_as_btn, file_picker, upload_button, save_template_btn, app_dd, load_assessment_dd, template_name_dd],
                ).then(fn=lambda _,y:y, 
                        inputs=[llm_provider, api_key], outputs=[api_key], js = """
                    (llm_provider, dummy) => { 
                        //try and retrieve the selected LLM provider (encrypted) API key from the browser's local storage if it exists already
                        const key_name = llm_provider.split(':')[0];
                        const key_value = localStorage.getItem(key_name); 
                        console.log('retrieved ', key_name, key_value); 
                        return [llm_provider, key_value || '']; 
                    }"""
                ).then(fn=set_model_key, inputs=[ss, llm_provider, api_key], outputs=[ss])

    return demo



if __name__ == "__main__":
    share = False
    #share = True
    from agent import SessionState as SessionState
    demo = ui(SessionState, app_name="Privacy Assessment", revision=None)
    demo.queue()
    demo.launch(share=share, ssr_mode=False)  #need to call it demo for gradio's reload mode
