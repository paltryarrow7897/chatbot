import ollama
import gradio as gr
from gradio import ChatMessage
from typing import List, Generator, Tuple
import os


model_name = "qwen2"


def store_history(
        user_message: str, 
        history: List[ChatMessage], 
        assistant_message: str
    ) -> List[ChatMessage]:
    if user_message:
        history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})
    return history


def clear_prompts() -> Tuple[gr.Textbox, gr.Textbox]:
    return gr.Textbox(value=None), gr.Textbox(value=None)


def chat(
        history: List[ChatMessage], 
        prompt: str
    ) -> Generator[str, str, str]:
    history.append({"role": "user", "content": prompt})
    stream = ollama.chat(
        model=model_name,
        messages=history,
        stream=True
    )
    output = ""
    for chunk in stream:
        output += chunk['message']['content']
        yield output


with gr.Blocks(title="OLLAMA | Qwen2 7B", theme=os.environ["GR_THEME_MIKU"]) as demo:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                history = gr.Chatbot(
                    label="Chat History", 
                    type="messages"
                )
                assistant_message = gr.Textbox(
                    label="Assistant Reply", 
                    interactive=False
                )
                prompt = gr.Textbox(
                    label="Your Message",
                    placeholder="Use ENTER to Chat, Shift+ENTER for line-break"
                )
                prompt.submit(
                    fn=chat, 
                    inputs=[history, prompt], 
                    outputs=assistant_message
                ).then(
                    fn=store_history, 
                    inputs=[prompt, history, assistant_message], 
                    outputs=history
                ).then(
                    fn=clear_prompts, 
                    inputs=None, 
                    outputs=[prompt, assistant_message]
                )


demo.launch(
    show_error=False,
    show_api=False
)