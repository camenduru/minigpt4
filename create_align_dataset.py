import argparse
import os
import json
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import webdataset as wds
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat

openai.api_key = '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'


def prepare_chatgpt_message(task_prompt, paragraph):
    messages = [{"role": "system", "content": task_prompt},
                {"role": "user", "content": paragraph}]
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt(chatgpt_messages, max_tokens=200, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=0.7, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens


def main(args):

    print('Initializing Chat')
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.device))

    ckpt_path = '/ibex/project/c2133/vicuna_ckpt_test/Vicuna_pretrain_stage2_cc/20230405233_3GPU40kSTEP_MAIN/checkpoint_3.pth'
    ckpt = torch.load(ckpt_path)
    msg = model.load_state_dict(ckpt['model'], strict=False)


    vis_processor_cfg = cfg.datasets_cfg.cc_combine.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    text_processor_cfg = cfg.datasets_cfg.laion.text_processor.train
    text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)

    chat = Chat(model, vis_processor, args.device)
    print('Initialization Finished')



    texts = {}
    negative_list = []

    for i in tqdm(range(args.begin_id, args.end_id)):
        image = Image.open(os.path.join(args.save_dir, 'image/{}.jpg'.format(i))).convert('RGB')

        fix_prompt = \
            "Fix the error in the given paragraph. " \
            "Remove any repeating sentences, meanless characters, not English sentences, and so on." \
            "Remove unnecessary repetition." \
            "Rewrite any incomplete sentences." \
            "Return directly the results WITHOUT explanation." \
            "Return directly the input paragraph if it is already correct WITHOUT explanation."

        answers = []
        answer_tokens = 0
        chat.reset()
        chat.upload_img(image)
        chat.ask("Describe this image in detail. Give as many details as possible. Say everything you see.")
        answer, tokens = chat.answer()
        answers.append(answer)
        answer_tokens += tokens
        if len(answer_tokens) < 80:
            chat.ask("Continue")
            answer, answer_token = chat.answer()
            answers.append(answer)
            answer_tokens += tokens
        answer = ' '.join(answers)

        chatgpt_message = prepare_chatgpt_message(fix_prompt, answer)
        improved_answer, num_token = call_chatgpt(chatgpt_message)

        if 'already correct' in improved_answer:
            if 'repetition' in improved_answer:
                continue
            improved_answer = answer
        if 'incomplete' in improved_answer or len(improved_answer) < 50:
            negative_list.append(improved_answer)
        else:
            texts[i] = improved_answer

    with open(os.path.join(args.save_dir, "cap_{}_{}.json".format(args.begin_id, args.end_id)), "w") as outfile:
        # write the dictionary to the file in JSON format
        json.dump(texts, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Alignment")

    parser.add_argument("--cfg-path", default='train_config/minigpt4_stage2_align.yaml')
    parser.add_argument("--save-dir", default="/ibex/project/c2133/blip_dataset/image_alignment")
    parser.add_argument("--begin-id", type=int)
    parser.add_argument("--end-id", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    print("begin_id: ", args.begin_id)
    print("end_id: ",  args.end_id)
    print("device:", args.device)

    main(args)
