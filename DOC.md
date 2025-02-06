# MAGI

This is yet another LLM-powered wannabe smart assistant.

See the `Pre-Requisites` section below to get started.

## Main Concepts

The starting premise is the now traditional 'chatbot' where you exchange messages back and forth with the assistant. These messages are displayed in the chat history window.

*The `Assitant Notes` companion document*. The novelty is that you are also sharing a document with the assistant. The format of the document is HTML (embedded javascript is not supported). Both you and the agent can edit the document. However you can restrict the parts that the agent can change (the agent can edit any HTML component that has an id=... attribute - including children). This allows you to provide an outline of the document you want to prepare; for example it could be a questionnaire that the agent must fill based on information it will get from you.

You can (and should) also include <class=Instructions> tags to provide further detailed instructions that the agent will have to follow. Having such instructions close to the text to be edited in the document can make the LLM more reliable than having a long list of instructions at the top of the conversation in a system prompt. Using the questionnaire example, you can include instructions in your outline to detail the kind of information the agent should obtain.

*Self-Critique*. The agent is also provided with a kind of Jiminy Cricket voice in its head, an ability to critique its own work. It will invoke this critique function from time to time to ensure that the conversation and companion documents remain aligned with your expectations. This can be useful to avoid LLM hallucinations among other things. In the case of the questionnaire example, the critique will make sure that all questions have been answered, and in a way that satisfies the instructions.

*Library*. It goes without saying that the assistant also comes with a (rudimentary) Retrieval Augmented Generation (RAG) capability. No self-respecting assistant can do without this nowadays. You can upload has many documents as you need into the library. The agent will search for needed information in this library to avoid having to ask you questions. Note that you can also upload a document directly in the chat message: this is different as such a document becomes part of the conversation and will be entirely passed to the LLM from then on. One one hand this ensures that the LLM reads the full document at each turn of the conversation, but on the other hand it consumes a lot of resources compared to RAG.

*Agency*. The assistant has 'agency' (albeit in a very limited manner: it is not going to make reservations at your favorite restaurant), in the sense that it will keep thinking (for example making edits, searching for information, making further edits, running a critique, making more edits, etc...) until it thinks that it has done all it could without your help. There is a safety limit built-in, but still it can consume quite a bit of your LLM credit. There is a stop button you can use.

*Rollback*. Like with other LLMs, the conversation may go in direction that does not work for you. There is a 'rollback' button to reset the conversation to a previous turn.

*Application Template*. We have not quite reach the Artificial General Intelligence (AGI) goal quite yet. We still need to adapt our LLM prompts to use-cases. The user interface makes it is possibe to craft, save and start new conversations from `application templates`. An app template comes with a LLM prompt that defines the agent's master instructions, an initial version of the assistant notes with the desired outline and specific instructions, another prompt for the Critique, and yet another prompt for the Librarian.

*Save a Conversation*. You can save a conversation and the companion document to a database. It can be public or private.

*Explainability*. LLMs are mysterious. But there is a screen to ask the LLM why it answered the way it did. The idea is that whatever triggered the LLM will also trigger when you prompt it with this question along with the entire conversation and prompts. This functionality is quite useful when designing your new app template: bootstrap with a first version your prompts and document instructions, run a conversation, then describe to the LLM the behavior in this conversation that you want to avoid and ask it to craft a better prompt. Repeat.

## Pre-Requisites

*HuggingFace Login*. This is free. It will be necessary for you to also get a user account at [Hugging Face](https://huggingface.co/join) in order to save documents in the library, conversations, or new application templates. You should do this before starting a conversation, as login-in has the undesired effect of clearing your conversation...

*LLM API Keys*. On the other hand, LLMs are not free! Someone needs to pay for the tokens they consume and produce, and that would be you. You need to get a subscription from a provider such as [OPENAI](https://openai.com/index/openai-api/), [ANTHROPIC](https://www.anthropic.com/pricing#anthropic-api) or [HUGGINGFACE](https://huggingface.co/join). Give them your credit card (when there is no teaser plan), set a budget and get an *API Key*. Then copy the API key into the assistant's screen: it will be saved (after being encrypted) as a cookie on your browser so that you need do this once only.

## Getting Started

The user interface has a:

- side bar panel where you can login, enter your API keys, start, load or save a conversation, and load a document into the library
you can hide / show this panel with the << / >> icon.

- a Chat tab with the chat conversation and the companion document. You can rollback the conversation, or interrupt the agent if it goes crazy on you.

- an App design tab, to edit an existing App Template or create new ones

- an Explain tab to ask the LLM to explain itself

### Basic use: 

You can start using the assistant once you have your API Key, and preferably your Hugging Face login.
1- Select an `application template` and start a `New Assessment`.
2- Optional: `Edit` the Assistant Notes document if you already have some sort of outline in mind or if you want to change the detailed instructions
3- Talk to the assistant...
4- Optional: upload documents to the library. You can do this at any time. Checking the 'App Default' option will make the document availale by default to all conversations based on the app template you have selected.
5- Save your conversation if you think it worth keeping. Checking the 'public' button will make it visible by everyone.

### App Designer

The App Design tab allows you to look under the hood and modify the parts. You can load an existing template and modify it. Versions of the templates are kept with 'revisions', so you can come back to a previous template. You can also save a template under a different name. 

An App Template consists of:

- the main prompt with the agent's master instructions

- the starting version of the Assistant Notes. This needs to be an HTML document. You should have at least one element with an id tag, like 
> <div id=0>...</div>
The numbering will be recomputed automatically, so do not worry about it.

- the Critique prompt, as well as the description of what the Critique tool does and the one parameter it accepts. Chaning the tool description and the parameter description will affect how the agent invokes the critique. 

- the Librarian prompt. It might be useful to provide specific context depending on your use-case.

### Explain

This screen can be used to ask the agent to explain itself. 

Select the specific interaction you are interested in by changing the LLM Call Number: you will see appear the corresponding message you got from the LLM for this specific interaction, along with details about the full details of the 'completion', and also the full prompt that was used by the LLM to generate this answer. 

Then below you can ask a question. Press the `Explain` button to run the LLM with all of that and get an answer to your question. 

The idea is that whatever piece of the model was trigger in the interaction you want to understand will also be triggered to answer the question, since most of the prompt is the same.



