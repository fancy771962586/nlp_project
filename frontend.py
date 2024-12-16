import json


from pydantic import BaseModel
import pandas as pd
import uvicorn
import gradio as gr
import threading
import re
import base64
import httpx
import asyncio
from loguru import logger
from utils import build_prompt, save_history, history_to_str
import time



# Gradio 聊天界面的异步响应函数
async def respond(message, chat_history):
    # 将用户标签和消息添加到聊天历史
    chat_history = chat_history + [{"role": "user", "content": "<div class='user-label'><b>User</b></div>\n<div class='message user'>" + message + "</div>"}]
    yield "", chat_history  # 立即显示用户消息
    m = save_history(chat_history, 4)
    history = history_to_str(m)
    logger.info("Current history:\n{}".format(history))
    # 显示“搜索中”消息，同时等待 10 秒
    loading_message = "<div style='display: flex; align-items: center;'>\n    <div class='loading-icon'></div>\n    <div style='font-size: 18px; color: #D3D3D3; text-align: center;'>Searching in progress...</div>\n</div>"
    chat_history = chat_history + [{"role": "assistant", "content": loading_message}]
    yield "", chat_history  # 显示加载消息

    search_retry = 0

    # 准备逐字发送机器人的响应
    async with (httpx.AsyncClient() as client):
        data = {"question": message}
        logger.debug("Searching the relevant documents...")
        while search_retry <=3:
            try:
                res = await client.post("http://127.0.0.1:8001/search", json=data, timeout=httpx.Timeout(20.0))
                context = res.json()['response']
                data, text1, text2 = build_prompt(context, history, message)
                logger.info("prompt:\n{}".format(text1 + text2))
                break
            except:
                search_retry += 1

        retries = 0
        while retries<=3:
            try:
                if retries!=0:
                    logger.warning(f'Retrying {retries}th times...')
                async with client.stream("POST", "http://127.0.0.1:8001/chat", json=data, timeout=httpx.Timeout(20.0)) as response:
                    if response.status_code == 200:
                        assistant_message = ""
                        # 移除加载消息，准备机器人的响应
                        chat_history[-1]["content"] = "<div class='assistant-label'><b>McSmartBot</b></div>\n<div class='message assistant'></div>"
                    # 逐字发送机器人的响应
                        async for word in response.aiter_bytes():
                            response_text_chunk = word.decode('utf-8')
                            assistant_message += response_text_chunk
                            # assistant_message = assistant_message.replace('\n\n', '<br><br>')
                            assistant_message = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', assistant_message)

                            chat_history[-1]["content"] = f"<div class='assistant-label'><b>McSmartBot</b></div>\n<div class='message assistant'>{assistant_message}</div>"
                            yield "", chat_history
                            await asyncio.sleep(0.05)  # Delay to simulate typing effect
                        logger.success(f"Generate response successfully, content:\n{chat_history[-1]['content']}")
                        break
                    else:
                        logger.error(f"{response.status_code}")
                        raise ValueError
            except httpx.TimeoutException:
                logger.warning(f"Timeout while waiting for response")
                retries += 1
                if retries <= 3:
                    error_message = f"<div style='display: flex; align-items: center;'>\n    <div class='loading-icon'></div>\n    <div style='font-size: 18px; color: #D3D3D3; text-align: center;'>Something wrong, retrying for the {retries}th time...</div>\n</div>"
                    chat_history[-1]["content"] = error_message
                    yield "", chat_history
            except Exception as e:
                logger.error(e)
                retries +=1
                if  retries<= 3:
                    error_message = f"<div style='display: flex; align-items: center;'>\n    <div class='loading-icon'></div>\n    <div style='font-size: 18px; color: #D3D3D3; text-align: center;'>Something wrong, retrying for the {retries}th time...</div>\n</div>"
                    chat_history[-1]["content"] = error_message
                    yield "", chat_history
        if retries>3:
            error_message = "<div class='assistant-label'><b>McSmartBot</b></div>\n<div class='message assistant'>Exceed max retries. Please reset the system by clicking `clear` button.</div>"
            chat_history[-1]["content"] = error_message
            yield "", chat_history



# 读取图像并转换为 base64 编码
with open("./src/logo.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# 定义带有 Blocks 的 Gradio 界面
def gradio_interface():
    custom_css = """
    /* 引入 Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Lobster&family=Poppins&display=swap');

    /* 用户和机器人标签 */
    .user-label, .assistant-label {
        font-weight: bold;
        margin-bottom: 5px;
        display: inline-block;
        margin-right: 10px;
    }

    .gradio-container {
        background-color: #FF8C00; /* 更改背景颜色 */
    }

    /* 修改聊天框的样式 */
    .custom-chatbot {
        background-color:  #FFD580;  /* 背景颜色：浅橙色 */
        border-radius: 10px;  /* 圆角 */
        padding: 20px;  /* 内边距 */
        max-height: 600px;  /* 最大高度 */
        overflow-y: auto;  /* 滚动条 */
        font-family: 'Arial', sans-serif;  /* 字体 */
        font-size: 14px;  /* 字体大小 */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  /* 阴影效果 */
        position: relative;  /* 使内部元素能够居中对齐 */
    }

    /* 用户消息样式 */
    .custom-chatbot .message.user {
        align-self: flex-end;  /* 右对齐 */
        max-width: none;        /* 移除最大宽度限制 */
        word-wrap: break-word;  /* 允许长单词换行 */
    }

    /* 机器人消息样式 */
    .custom-chatbot .message.assistant {
        align-self: flex-start;  /* 左对齐 */
        max-width: none;          /* 移除最大宽度限制 */
        word-wrap: break-word;  /* 允许长单词换行 */
    }

    /* 修改输入框的样式 */
    .custom-textbox  {
        border: 2px solid #ccc;
        border-radius: 8px;
        padding: 4px;          /* 控制输入框内部的填充 */
        font-size: 16px;
        color: #333;
        width: auto;           /* 设置宽度为自动，避免填满整个行 */
        max-width: 70%;        /* 限制输入框的最大宽度 */
        box-sizing: border-box; /* 确保 padding 不会增加输入框的整体宽度 */
    }

    /* 修改提交按钮的样式 */
    .custom-submit-btn, .custom-clear-btn {
        background-color: #FFB84D; /* 按钮背景色 */
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 12px 0px;
        border: none;
        cursor: pointer;
    }

    .custom-submit-btn:hover, .custom-clear-btn:hover {
        background-color: #FFD580; /* 按钮悬停颜色 */
    }

    footer {
        display: none !important;
    }

    .usage-link {
        display: none !important;
    }

    /* 示例按钮的颜色调整：*/
    /* 修改示例按钮的背景颜色为白色，悬停时变为浅灰色，并添加过渡效果。*/
    .example-box {
        background-color: #FFFFFF;  /* 示例框的白色背景 */
        ...
        transition: background-color 0.3s ease; /* 添加过渡效果 */
    }
    
    .example-box:hover {
        background-color: #E0E0E0; /* 示例框悬停时颜色，变成浅灰色 */
    }

    /* “Searching in progress”动画的加载图标 */
    .loading-icon {
        width: 20px;
        height: 20px;
        border: 3px solid #D3D3D3;
        border-top: 3px solid #FF8C00;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* 设置标题的字体为 Lobster */
    .title-text {
        font-family: 'Lobster', cursive;
        margin: 0;
    }

    /* 设置副标题的字体为 Poppins */
    .subtitle-text {
        font-family: 'Poppins', sans-serif;
        margin: 0;
    }
    """
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown(f"""
        <div style="display: flex; align-items: center; justify-content: center;">
            <img src="data:image/jpg;base64,{base64_image}" alt="Logo" style="width: 80px; height: 80px; margin-left: -80px;margin-right: 50px;">
            <div>
                <h1 class="title-text">McSmartBite</h1>
                <h4 class="subtitle-text">Eat Smart, Live Smart with McSmartBite!</h4>
            </div>
        </div>
         <script>
            document.addEventListener("DOMContentLoaded", function() {{
                document.querySelectorAll('footer, .usage-link').forEach(el => el.style.display = 'none');
            }});
        </script>
        """)

        chatbot = gr.Chatbot(value=[], type="messages", elem_classes=["custom-chatbot"])

        with gr.Row():
            msg = gr.Textbox(placeholder="You can ask me anything about McDonald's.", container=False, scale=12, elem_classes=["custom-textbox"])
            submit_btn = gr.Button("Submit", elem_classes=["custom-submit-btn"], scale=1)
            clear = gr.ClearButton([msg, chatbot], elem_classes=["custom-clear-btn"], scale=1)

        # 使用异步函数进行响应
        msg.submit(respond, [msg, chatbot], [msg, chatbot], concurrency_limit=5)
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot], concurrency_limit=5)  # 将提交按钮点击事件链接到 respond 函数
        clear.click(lambda: None, None, chatbot, queue=False)

        # 添加可点击的示例按钮，并自动提交
        examples = [
            "My budget is 50 RMB and I like fried chicken, what can I order?",
            "What is the most popular meal at McDonald's?",
            "How many McDonald's stores are there in China?"
        ]



        gr.Markdown("""
        <div style="margin-top: 20px; font-size: 1rem; color: #555;">
            <i style="margin-right: 8px; font-style: normal; font-size: 1.2rem;">&#128161;</i>
            <span>You may wanna ask:</span>
        </div>
        """)

        with gr.Row():
            for example in examples:
                gr.Button(example, elem_classes=["example-box"]).click(
                    lambda e=example: (e, gr.update()), None, [msg, chatbot]
                ,concurrency_limit=5).then(
                    fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot]
                )


    return demo

# 启动 Gradio 界面
# gradio_interface().launch(share=True, server_name='0.0.0.0', server_port=7860, auth=("linguistic", "lions"))
gradio_interface().launch(share=True, server_name='0.0.0.0', server_port=7860)
