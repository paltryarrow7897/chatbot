import gradio as gr
from gemini_utils import chat, GEMINI_FILE_TYPES
from dotenv import load_dotenv
import os


load_dotenv(override=True)


def clear_chat_query() -> gr.Textbox:
    return gr.Textbox(value=None)


def issue_info():
    return gr.Info(message="Please wait until Status changes to ACTIVE")


with gr.Blocks(
    title="LLM Interactive Showcase", 
    theme=os.environ["GR_THEME_MIKU"]
    # theme="JohnSmith9982/small_and_pretty"
) as demo:
    
    with gr.Row():
        gr.Markdown("### 🛑IMPORTANT🛑: Do NOT upload potentially sensitive information.")

    with gr.Tab(label="Chat"):
        with gr.Column():
            gr.Markdown("#### Choose a model, and start chatting") 
            chat_model_name = gr.Radio(
                show_label=False, 
                choices=["gemini-2.5-pro-preview-03-25", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"], 
                value="gemini-2.0-flash"
            )
            with gr.Group():
                chat_history = gr.Chatbot(
                    show_label=False, 
                    type="messages", 
                    show_copy_all_button=True
                )
                chat_query = gr.MultimodalTextbox(
                    show_label=False, 
                    file_count="multiple", 
                    file_types=["image", "audio", "video"] + GEMINI_FILE_TYPES, 
                    placeholder="Press ENTER to chat, Shift+ENTER for line break"
                )
        chat_query.submit(
            fn=chat,
            inputs=[chat_model_name, chat_history, chat_query], 
            outputs=chat_history
        ).then(
            fn=clear_chat_query, 
            inputs=None, 
            outputs=chat_query
        )

    with gr.Row():
        gr.Markdown("### Powered by [Gemini API](https://ai.google.dev/)")

demo.queue().launch(
    show_api=False, 
    show_error=False, 
    root_path="/llm-demo", 
    server_port=7860,
    ssl_certfile="fullchain.pem",
    ssl_keyfile="privkey.pem",
    ssl_verify=False
)
