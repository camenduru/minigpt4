import gradio as gr


def handle_error(msg):
    return gr.outputs.HTML(f"<p style='color:red;'>Error: {msg}</p>")
