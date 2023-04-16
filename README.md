# MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models
[Deyao Zhu](https://tsutikgiau.github.io/)* (On Job Market!), [Jun Chen](https://junchen14.github.io/)* (On Job Market!), Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. *Equal Contribution

**King Abdullah University of Science and Technology**

[[Project Website]](https://minigpt-4.github.io/) [[Paper]](MiniGPT_4.pdf) [Online Demo]


## Online Demo

Chat with MiniGPT-4 around your images


## Examples
  |   |   |
:-------------------------:|:-------------------------:
![find wild](examples/wop_2.png) |  ![write story](examples/ad_2.png)
![solve problem](examples/fix_1.png)  |  ![write Poem](examples/rhyme_1.png)





## Abstract
The recent GPT-4 has demonstrated extraordinary multi-modal abilities, such as directly generating websites from handwritten text and identifying humorous elements within images. These features are rarely observed in previous vision-language models. We believe the primary reason for GPT-4's advanced multi-modal generation capabilities lies in the utilization of a more advanced large language model (LLM). To examine this phenomenon, we present MiniGPT-4, which aligns a frozen visual encoder with a frozen LLM, Vicuna, using just one projection layer. 
Our findings reveal that MiniGPT-4 processes many capabilities similar to those exhibited by GPT-4 like detailed image description generation and website creation from hand-written drafts. Furthermore, we also observe other emerging capabilities in MiniGPT-4, including writing stories and poems inspired by given images, providing solutions to problems shown in images, teaching users how to cook based on food photos, etc. 
These advanced capabilities can be attributed to the use of a more advanced large language model.
Furthermore, our method is computationally efficient, as we only train a projection layer using roughly 5 million aligned image-text pairs and an additional 3,500 carefully curated high-quality pairs. 








## Getting Started
### Installation

1. Prepare the code and the environment

Git clone our repository, creating a python environment and ativate it via the following command

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```


2. Prepare the pretrained Vicuna weights

The current version of MiniGPT-4 is built on the v0 versoin of Vicuna-13B.
Please refer to their instructions [here](https://huggingface.co/lmsys/vicuna-13b-delta-v0) to obtaining the weights.
The final weights would be in a single folder with the following structure:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

Then, set the path to the vicuna weight in the model config file 
[here](minigpt4/configs/models/minigpt4.yaml#L21) at Line 21.

3. Prepare the pretrained MiniGPT-4 checkpoint

To play with our pretrained model, download the pretrained checkpoint 
[here](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link).
Then, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigpt4.yaml](eval_configs/minigpt4.yaml#L15) at Line 15. 





### Launching Demo Locally

Try out our demo [demo.py](app.py) with your images for on your local machine by running

```
python demo.py --cfg-path eval_configs/minigpt4.yaml
```





### Training
The training of MiniGPT-4 contains two-stage alignments.
In the first stage, the model is trained using image-text pairs from Laion and CC datasets
to align the vision and language model. To download and prepare the datasets, please check 
[here](dataset/readme.md). 
After the first stage, the visual features are mapped and can be understood by the language
model.
To launch the first stage training, run 

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_config/minigpt4_stage1_laion.yaml
```

In the second stage, we use a small high quality image-text pair dataset created by ourselves
and convert it to a conversation format to further align MiniGPT-4.
Our second stage dataset can be download from 
[here](https://drive.google.com/file/d/1RnS0mQJj8YU0E--sfH08scu5-ALxzLNj/view?usp=share_link).
After the second stage alignment, MiniGPT-4 is able to talk about the image in
a smooth way. 
To launch the second stage alignment, run

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_config/minigpt4_stage2_align.yaml
```





## Acknowledgement

+ [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
+ [Vicuna](https://github.com/lm-sys/FastChat)


If you're using MiniGPT-4 in your research or applications, please cite using this BibTeX:
```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing the Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License
This repository is built on [Lavis](https://github.com/salesforce/LAVIS) with BSD 3-Clause License
[BSD 3-Clause License](LICENSE.txt)
