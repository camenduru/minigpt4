import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from minigpt4.common.dist_utils import get_rank
import csv


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def preprocess_image(image):
    # Resize the image to a fixed size
    image = image.resize((224, 224))

    # Normalize the image
    image = np.array(image) / 255.0
    image = (image - 0.5) / 0.5

    return image


def create_csv_file(chatbot, img_names, filename="chat_output.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Question', 'Answer'])
        for i in range(len(chatbot)):
            writer.writerow([img_names[i], chatbot[i][0], chatbot[i][1]])
    return filename
