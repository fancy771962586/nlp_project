import json
from datetime import time
from typing import List
import requests
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import gradio as gr
import base64
import httpx
import asyncio
import time
from loguru import logger
from utils import  save_history, history_to_str, build_prompt





class Query(BaseModel):
    budget: float = None
    max_calories: float = None
    user_message: str


async def respond(message, chat_history):
    # Add User label above the user message
    user_message = f"<div class='label'><b>User</b></div><div class='message'>{message}</div>"
    chat_history.append({"role": "user", "content": user_message})
    logger.info("User Query:{}".format(message))
    yield "", chat_history  # Yield the chat history to display the user message right away

    # Prepare the assistant's response
    async with httpx.AsyncClient() as client:
            assistant_label = "<div class='label'><b>McSmartBot</b></div>"
            chat_history.append({"role": "assistant", "content": assistant_label})  # Add assistant label
            print("chat:",chat_history)
            m = save_history(chat_history, 4)
            history = history_to_str(m)
            print(history)
            logger.info("Current history:\n{}".format(history))
            data = {"question": message}
            logger.debug("Searching the relevant documents...")
            res = requests.post("http://127.0.0.1:8001/search",  json=data)
            assistant_message = "SEARCHING..."
            chat_history[-1]["content"] = f"{assistant_label}<div class='message'>{assistant_message}</div>"
            yield "", chat_history
            context = json.loads(res.content.decode())['response']
            logger.info("searched context:\n",context)
            data, text = build_prompt(context,history,message)
            logger.info("prompt:\n{}".format(text))
            # response = requests.post("http://127.0.0.1:8001/chat", stream=True, json=data)
            retries = 0
            while retries <= 3:
                try:
                    async with client.stream("POST","http://127.0.0.1:8001/chat",json=data,timeout=10.0) as response:
                        # async for chunk in response.aiter_bytes():
                        #         print(chunk.decode('utf-8'))

                            if response.status_code == 200:
                                assistant_message = ""
                                # for word in response.iter_content(chunk_size=2048, decode_unicode=True):
                                async for word in response.aiter_bytes():
                                        assistant_message += word.decode('utf-8')
                                        chat_history[-1]["content"] = f"{assistant_label}<div class='message'>{assistant_message.strip()}</div>"
                                        yield "", chat_history
                                        print(chat_history)
                                        await asyncio.sleep(0.05)  # Delay to simulate typing effect
                                break


                            else:
                                error_message = "<div class='label'><b>McSmartBot</b></div><div class='message'>Error: Unable to fetch response.</div>"
                                chat_history.append({"role": "assistant", "content": error_message})
                                print(chat_history)
                                yield "", chat_history
                                raise HTTPException(status_code=500)
                except:
                    logger.warning("Request time out, retrying {} time(s)".format(retries))
                    retries += 1
            if retries == 3:
                logger.error("Max retries exceeded")



# Read the image and convert to base64 encoding
with open("C:/Users/Administrator.DESKTOP-LTJNBTE/Desktop/麻衣.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Define a Gradio interface with Blocks
def gradio_interface():
    custom_css = """
    /* Custom CSS for labels and chat messages */
    .label {
        font-weight: bold;
        margin-bottom: 2px;
        font-size: 1em;
    }

    .message {
        background-color: #FFF;
        padding: 10px;
        border-radius: 10px;
        font-size: 0.9em;
        box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.1);
    }

    .gradio-container {
        background-color: #FF8C00;
    }

    .custom-chatbot {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        max-height: 600px;
        overflow-y: auto;
        font-family: 'Arial', sans-serif;
        font-size: 14px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Chat message alignment */
    .custom-chatbot .message.user {
        align-self: flex-end;
    }

    .custom-chatbot .message.assistant {
        align-self: flex-start;
    }

    /* Textbox styling */
    .custom-textbox {
        border: 2px solid #ccc;
        border-radius: 8px;
        padding: 4px;
        font-size: 16px;
        color: #333;
        max-width: 70%;
    }

    /* Button styling */
    .custom-submit-btn, .custom-clear-btn {
        background-color: #FFB84D;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 12px 0px;
        border: none;
        cursor: pointer;
    }

    .custom-submit-btn:hover, .custom-clear-btn:hover {
        background-color: #FFD580;
    }

    footer {
        display: none !important;
    }

    .usage-link {
        display: none !important;
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown(f"""
        <div style="display: flex; align-items: center; justify-content: center;">
            <img src="data:image/jpg;base64,{base64_image}" alt="Logo" style="width: 80px; height: 80px; margin-left: -80px;margin-right: 50px;">
            <div>
                <h1 class="center-title">McSmartBite</h1>
                <h4 class="center-description">Eat Smart, Live Smart with McSmartBite!</h4>
            </div>
        </div>
        """)

        chatbot = gr.Chatbot(value=[], type="messages", elem_classes=["custom-chatbot"])
        with gr.Row():
            msg = gr.Textbox(placeholder="Please provide your calories and budget.", container=False, scale=12, elem_classes=["custom-textbox"])
            submit_btn = gr.Button("Submit", elem_classes=["custom-submit-btn"], scale=1)
            clear = gr.ClearButton([msg, chatbot], elem_classes=["custom-clear-btn"], scale=1)

        # Use async function for response
        msg.submit(respond, [msg, chatbot], [msg, chatbot],concurrency_limit=5)
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot],concurrency_limit=5)
        clear.click(lambda: None, None, chatbot, queue=False)

    return demo




# Launch the Gradio interface
gradio_interface().launch(share=True)



