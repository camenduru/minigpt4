import os
import csv
import gradio as gr
from PIL import Image
import json

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# Imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# Importing functions from the new modules
from config_parser import parse_args
from utils import setup_seeds, preprocess_image
from error_handler import handle_error
from gradio_interface import gradio_reset, upload_img, gradio_ask, gradio_answer
from model_setup import initialize_model

SHARED_UI_WARNING = f'''### [NOTE] This is still in beta, problems may occur.
'''

title = """<h1 align="center">Demo of MiniGPT-4</h1>"""
description = """<h3>This is the demo of MiniGPT-4, brought to you by Wadah Adlan and the team. Upload your images and start chatting!</h3>"""
article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://github.com/TsuTikgiau/blip2-llm/blob/release_prepare/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p><p><b>Special thanks to Wadah Adlan for his valuable contributions to the MiniGPT-4 project.</b></p>
"""


def main():
    # Initialize Chat
    print('Initializing Chat')
    cfg, model, chat = initialize_model(parse_args())

    setup_seeds(cfg)

    # Gradio Interface
    with gr.Interface(
        fn=gradio_answer,
        inputs=[
            gr.inputs.Textbox(
                label='User', placeholder='Please upload your images first', interactive=False),
            gr.inputs.Image(type="pil", label="Upload Images", multiple=True),
            gr.inputs.Slider(
                minimum=1,
                maximum=5,
                default=1,
                step=1,
                label="Beam Search Numbers",
                help="The number of beams used during decoding."
            ),
            gr.inputs.Slider(
                minimum=0.1,
                maximum=2.0,
                default=1.0,
                step=0.1,
                label="Temperature",
                help="The temperature value used during decoding."
            ),
        ],
        outputs=[
            gr.outputs.Textbox(label='Chatbot'),
            gr.outputs.HTML(label='UI Warning')
        ],
        title=title,
        description=description,
        article=article,
        analytics_enabled=False,
        examples=[
            ['', '', 1, 1]
        ],
        allow_flagging=False,
        allow_screenshot=False,
        theme="huggingface",
    ).launch(inline=False, debug=True):
        print('Gradio is up...')


if __name__ == "__main__":
    main()
