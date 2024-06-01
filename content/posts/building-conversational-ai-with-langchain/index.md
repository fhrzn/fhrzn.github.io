---
# title: 'Build an AI Chatbot with OpenRouter and Langchain'
title: 'Building Conversational AI with LangChain: Techniques for Context Retention in Chatbots'
date: 2024-05-26T00:34:59+07:00
tags: ["chatbot", "conversational ai", "context aware", "langchain", "rag"]
draft: false
description: ""
disableShare: true
hideSummary: false
ShowReadingTime: true
ShowWordCount: true
cover:
    image: "cover.jpg" # image path/url
    alt: "Cover Post" # alt text
    caption: "Photo by [Aaron Jones](https://unsplash.com/@ajonesyyyyy?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/person-riding-bicycle-on-road-during-daytime-i2MYu4AElsE?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash)" # display caption under cover
    relative: true # when using page bundles set this to true
math: katex
keywords: ["chatbot", "conversational ai", "context aware", "langchain", "rag"]
summary: "We moved from in-house training model to hosted models and ready-to-use APIs. With the existence of free LLM APIs, let's explore how to create our own free chatbot!"
---

Since the rise of LLMs era such as ChatGPT, Bard, Gemini, Claude, etc. the development of AI based application has been drastically changed. We moved from the conventional in-house training (which require high machine spec) to ready-to-use APIs.

Fortunately, there are free APIs curated on OpenRouter platform. So we can build our simple chatbot without spending a penny!

Let's get started! 🔥

>⚡️ If you want to jump directly to the repository, you can access it [here](https://github.com/fhrzn/study-archive/tree/master/simple-rag-openrouter).

## Project Brief
Before we start working, let's take a look on high level concept below of how it will works.
![Conversational AI Architecture](images/convai_arch.png#center)

For those who are just started learning LLM, this brief information can help you to understand more how LLM-based application works.

Basically, LLM works by taking user input and answer them based on its internal knowledge. If we want our LLM to do specific task such as brainstorming, making travel plan, calculating our expenses, or etc. we will need more advanced and structured user input. This is what we called **Prompt**.

Here are very simple illustrations of the difference in user input structure when adding a prompt and not.
```
Human: <user query/input here>
Assistant: <chatbot answer here>
```

```
System Prompt: <system prompt here>
Human: <user query/input here>
Assistant: <chatbot answer here>
```

At this point, our LLM should be able to do specific and more advanced task. However, the LLM doesn't remember the prior conversation and every time we invoke LLM call it assume that current user input as initial conversation with the user. To solve this problem, we need to add our previous conversation to the prompt. So that everytime we send user input, the LLM has knowledge of prior conversation which makes it *remember* previous conversation contexes. At the end of conversation, right after the LLM given its responses, we need to save both user input and LLM response.

Here is the simple illustration of our system prompt which incorporating previous conversation.
```
You are professional travel planner. You are able to convert different timezone to the desired timezone quick and precisely.
...
...
---------------------
Chat history:
{chat_history}
```

Pay attention that we are giving clear separation between system prompt (act as basic command) and previous chat history (as additional knowledge/context). And we put `chat_history` variable in curly braces that intended to be replaced with our retrieved previous conversation later. We will talk about it more in technical implementation.

In short, our chatbot will combine both user input and previous chat history in a prompt. Then, it will passed to LLM as the input. The LLM then responsible to generate answer based on given input. Finally, we save current conversation (user input and LLM response) as chat history which will be consumed again later.


## Setup OpenRouter API Key
Mostly, each LLM has different APIs to the others. That makes switching between models become less straightforward. OpenRouter lift those problem for us by providing a unified interface which allow us to change between models easily with very minimal code changes.

Now, to get started make sure you already created an account in [https://openrouter.ai/](https://openrouter.ai/). Then, go to [Keys page](https://openrouter.ai/keys) and create your API Key. Remember to save it somewhere save as it won't show twice.
![OpenRouter API Key page](images/api_page.png#center)

## Start Coding! 🔥
### Setup Environment
In python, it is advised to create individual virtualenv to isolate our libraries. This can reduce the possibility of error due to conflict on library versions. We will use default python's `virtualenv` to make one. 

Run the commands below on terminal
```bash
mkdir simple-rag-openrouter && cd simple-rag-openrouter
python -m venv venv
source .venv/bin/activate
```

After running the commands above, a new folder named `venv` should be appeared in our project directory. All of our installed library will be saved there.

### Install Libraries
Make sure you already activate the virtualenv. Then, run this command to install required libraries.

```bash
pip install langchain langchain_openai gradio langchain_community uuid httpx
```

### Create Project Environment Variables
Now, create a new file named `.env`. We will store our openrouter api key and base url there. So we can ensure our published codes didn't contains any confidential information.

You also may create the file using terminal
```bash
touch .env
```

Then, open the `.env` file we just created and put our credentials there.
```
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = "<your-openrouter-api-key>"
```

To make our code reproducable by other person, let's dump our installed python libraries to a file called `requirements.txt`.
```bash
pip freeze > requirements.txt
```

### Building our Chatbot Interface
There are a various options to build chat UI, but here we will use gradio's `ChatInterface` which very handy to use.

Let's create a python file called `main.py` and put the codes below.
```python
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

def predict_chat():
    # TODO: we will put our LLM call here later
    pass

with gr.Blocks(fill_height=True) as demo:

    chat_window = gr.Chatbot(bubble_full_width=False, render=False, scale=1)

    chat = gr.ChatInterface(
        predict_chat,
        chatbot=chat_window,
        fill_height=True,
        retry_btn=None,
        undo_btn=None,
        clear_btn=None
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
```

To run our script, run the following command in terminal.
```bash
gradio main.py
```

And we can already see the chat interface provided by gradio.
![Gradio default ChatInterface](images/gradio_1.png#center)

Now, since openrouter provide a lot of LLM models we can choose and switch between them every time we want to send message. Think of the LLM here as replacable module that we can change and set to any model we want.
![Replacable LLM module](images/modular_llm.png#center)

Let's add all available openrouter models, but limited to the free version only as we dont want to spend any money. To get the full list of available models, we can perform API request to openrouter's endpoint here [https://openrouter.ai/api/v1/models](https://openrouter.ai/api/v1/models). Then we can put the available models as dropdown options above the chat interface.

First, create a new function to get all available free models.
```python3
import httpx

def get_free_models():
    res = httpx.get("https://openrouter.ai/api/v1/models")
    if res:
        res = res.json()
        # filter only free models
        models = [item["id"] for item in res["data"] if "free" in item["id"]]
        return sorted(models)
```

Then, inside add dropdown component and populate the model names.
```python3
with gr.Blocks(fill_height=True) as demo:

    models = get_free_models()      # get model names
    model_choice = gr.Dropdown(
        choices=models,             # populate model names
        show_label=True,
        label="Model Choice",
        interactive=True,
        value=models[0]
    )
```

Finally, add the newly added component as Chat Interface additional inputs.
```python3
chat = gr.ChatInterface(
    predict_chat,
    chatbot=chat_window,
    additional_inputs=[model_choice],
    fill_height=True,
    retry_btn=None,
    undo_btn=None,
    clear_btn=None
)
```

Our `main.py` file should be look like this.
```python3
import gradio as gr
from dotenv import load_dotenv
import httpx

load_dotenv()

def predict_chat():
    # TODO: we will put our LLM call here later
    pass


def get_free_models():
    res = httpx.get("https://openrouter.ai/api/v1/models")
    if res:
        res = res.json()
        # filter only free models
        models = [item["id"] for item in res["data"] if "free" in item["id"]]
        return sorted(models)


with gr.Blocks(fill_height=True) as demo:

    models = get_free_models()
    model_choice = gr.Dropdown(
        choices=models,             # populate model names
        show_label=True,
        label="Model Choice",
        interactive=True,
        value=models[0]
    )

    chat_window = gr.Chatbot(bubble_full_width=False, render=False, scale=1)

    chat = gr.ChatInterface(
        predict_chat,
        chatbot=chat_window,
        additional_inputs=[model_choice],
        fill_height=True,
        retry_btn=None,
        undo_btn=None,
        clear_btn=None
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
```

The code above should add dropdown list with free models name available in OpenRouter.
![Chat UI with model choices](images/chatui_w_choice.png#center)

Now we have all necessary components for our UI. However, we still can't play around with the LLMs as we haven't put logic to handle user-chatbot interactions yet. 

Let's add some logic there!

### Add Chatbot Logic
Remember that we've created a function called `predict_chat()` earlier? Now, to make the code cleaner we will move it to a new file named `chatbot.py`. We will put everything related to chatbot logic there including manage the conversation history.

We will use the prompt below to give specific task to the LLM.
> System prompt:
>
> *You are an AI assistant that capable to interact with users using friendly tone. Whenever you think it needed, add some emojis to your response. No need to use hashtags.*

Let's write some codes to file `chatbot.py`. Don't forget to create one if you don't have it yet.

First, let's create a prompt template to put our prompt and user input together.
```python3
from langchain.prompts import ChatPromptTemplate


def predict_chat(message: str, history: list, model_name: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that capable to interact with users using friendly tone. Whenever you think it needed, add some emojis to your response. No need to use hashtags."),
        ("human", "{input}")
    ])
```
Here, we are using `ChatPromptTemplate` as we want to format the prompt in the conversation style. There are only 3 roles available, system, human, and AI.

The `predict_chat()` function takes 3 input, namely message, history, and model_name. Both message and history is required by default as we used gradio's `ChatInterface`. While the model_name came from the model names dropdown in the `main.py` file.

>💡 If you're curious how in the world those dropdown input automatically required in the `predict_chat` function, it is because we put that component into `additional_inputs` in Chat Interface parameter.

Next, let's initiate our LLM instance. Then chain our prompt and LLM together.
```python3
llm = ChatOpenAI(
    model=model_name,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE")
)

# chain prompt and LLM instance using LCEL
chain = prompt | llm
```
Notice that we chain prompt and LLM together using pipe (`|`) symbol. Thanks to LangChain Expression Language (LCEL) we can write this handy shorthand.

Finally, we invoke the chain in stream mode so we can see the progressive output while it generates the full response.
```python3
partial_msg = ""
# for chunk in history_runnable.stream({"input": message}, config={"configurable": {"session_id": user_id}, "callbacks": [ConsoleCallbackHandler()]}):
for chunk in chain.stream({"input": message}):
    partial_msg = partial_msg + chunk.content
    yield partial_msg
```

Our `chatbot.py` should be look like this now.
```python3
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os


def predict_chat(message: str, history: list, model_name: str):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that capable to interact with users using friendly tone. Whenever you think it needed, add some emojis to your response. No need to use hashtags."),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE")
    )
    
    chain = prompt | llm

    partial_msg = ""
    # for chunk in history_runnable.stream({"input": message}, config={"configurable": {"session_id": user_id}, "callbacks": [ConsoleCallbackHandler()]}):
    for chunk in chain.stream({"input": message}):
        partial_msg = partial_msg + chunk.content
        yield partial_msg
```

Now, go back for a while to `main.py` and add this line on import section.
```python3
from chatbot import predict_chat
```

Cool! At this point, we can start playing around with our chatbot and it will respond to our chats!

![Playing around with our chatbot](images/chatui_demo.gif#center)

### Context-Aware Response Generation
Our chatbot can respond to our question already. However, it lack of previous conversation contexes. Take a look on the captured conversation below. The previous context was talking about **travel plan**, but when I tried to continue the conversation it gave me an answer that doesn't have correlation with previous context.
![Chatbot failed to understand prevoius context](images/chatui_fail_context.png#center)

To work on this issue, we need to put the chat history to our prompt. Here we will use SQLite as our database to save the whole chat history. Since we will only have one database for all users, we need a `session_id` between each user conversation history to avoid retrieving wrong user's conversation.

We will first add a hidden input in chat interface that generate a unique UUID which will act as our session_id. So, everytime we refresh the page, it will generate new session_id as well.

```python3
import uuid

with gr.Blocks(fill_height=True) as demo:

    user_ids = gr.Textbox(visible=False, value=uuid.uuid4())
```

Next, add the hidden input component as Chat Interface additional_inputs as well. So, now Chat Interface additional inputs should contains model_choice and user_ids. Otherwise, we cannot pass the value to the function `predict_chat()` behind.
```python3
chat = gr.ChatInterface(
    predict_chat,
    chatbot=chat_window,
    additional_inputs=[model_choice, user_ids],
    fill_height=True,
    retry_btn=None,
    undo_btn=None,
    clear_btn=None
)
```


Our final `main.py` should be look like this.
```python3
import gradio as gr
from dotenv import load_dotenv
import httpx
from chat import predict_chat
import uuid

load_dotenv()


def get_free_models():
    res = httpx.get("https://openrouter.ai/api/v1/models")
    if res:
        res = res.json()
        models = [item["id"] for item in res["data"] if "free" in item["id"]]
        return sorted(models)


with gr.Blocks(fill_height=True) as demo:

    user_ids = gr.Textbox(visible=False, value=uuid.uuid4())

    models = get_free_models()
    model_choice = gr.Dropdown(
        choices=models,
        show_label=True,
        label="Model Choice",
        interactive=True,
        value=models[0]
    )

    chat_window = gr.Chatbot(bubble_full_width=False, render=False, scale=1)

    chat = gr.ChatInterface(
        predict_chat,
        chatbot=chat_window,
        additional_inputs=[model_choice, user_ids],
        fill_height=True,
        retry_btn=None,
        undo_btn=None,
        clear_btn=None
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
```


Now, we're done with `main.py`. Let's move further to `chatbot.py` file.
```python3
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_chat_history(session_id: str):
    chat_history = SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///memory.db")
    return chat_history
```

Then, we'll a new variable named `history` in our prompt using `MessagesPlaceholder`. The rest of the prompt attribute stays the same. Also, don't forget to add user_id to our predict_chat function parameter
```python3
def predict_chat(message: str, history: list, model_name: str, user_id: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that capable to interact with users using friendly tone. Whenever you think it needed, add some emojis to your response. No need to use hashtags."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])
```

Next, instead of we call invoke directly on chained prompt and LLM instance, we will use a new instance called `RunnableWithMessageHistory`.
```python3
from langchain_core.runnables.history import RunnableWithMessageHistory

history_runnable = RunnableWithMessageHistory(
    chain,
    get_session_history=get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)
```
Remember that we should always save our current conversation to database so we can use it in future interaction with chatbot? Gratefully, all those logic already handled by `RunnableWithMessageHistory` so we don't have to handle it by ourselves.

Note that we also put `input` and `history` as input and history message key respectively. Keep in mind that this key should match with variables key that already defined in prompt.

Finally, rather than calling stream from `chain`, we call it from `history_runnable` instead. So our streaming code will look like this.
```python3
partial_msg = ""
for chunk in history_runnable.stream({"input": message}, config={"configurable": {"session_id": user_id}}):
    partial_msg = partial_msg + chunk.content
    yield partial_msg
```

Our final `main.py` should be look like this.
```python3
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory


def get_chat_history(session_id: str):
    chat_history = SQLChatMessageHistory(session_id=session_id, connection_string="sqlite:///memory.db")
    return chat_history


def predict_chat(message: str, history: list, model_name: str, user_id: str):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that capable to interact with users using friendly tone. Whenever you think it needed, add some emojis to your response. No need to use hashtags."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE")
    )

    chain = prompt | llm

    history_runnable = RunnableWithMessageHistory(
        chain,
        get_session_history=get_chat_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    partial_msg = ""
    for chunk in history_runnable.stream({"input": message}, config={"configurable": {"session_id": user_id}}):
        partial_msg = partial_msg + chunk.content
        yield partial_msg
```

That's a wrap! Our chatbot now can understand previous conversation context. Super! ⚡️
![Final chatbot that understand context from previous conversation](images/chatui_demo_final.gif#center)

### Full Project
[https://github.com/fhrzn/study-archive/tree/master/simple-rag-openrouter](https://github.com/fhrzn/study-archive/tree/master/simple-rag-openrouter)


## Conclusion
Throughout this article, we already covered how to build a context-aware chatbot -- a chatbot that understand previous conversation contexes.

In the upcoming article I will demonstrate how we can extend this chatbot to be able interact with external files as well such as financial reports, product catalogs, or even company profile website.

Stay tune! 👋

## References
1. [LangChain getting started](https://python.langchain.com/v0.1/docs/expression_language/get_started/)
2. [Add message history (memory)](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/)
3. [LCEL (LangChain Expression Language)](https://python.langchain.com/v0.1/docs/expression_language/why/)
4. [OpenRouter docs](https://openrouter.ai/docs/quick-start)

---

## Let's get Connected 🙌
If you have any inquiries, comments, suggestions, or critics please don’t hesitate to reach me out:

- Mail: affahrizain@gmail.com
- LinkedIn: https://www.linkedin.com/in/fahrizainn/
- GitHub: https://github.com/fhrzn



