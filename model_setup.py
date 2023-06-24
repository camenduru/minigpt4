import torch
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
from config_parser import parse_args


def initialize_model():
    print('Initializing Chat')

    # Parse arguments for configuration
    cfg = Config(parse_args())

    # Model configuration and initialization
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

    # Visual processor configuration and initialization
    vis_processor_cfg = cfg.datasets_cfg.cc_align.vis_processor.train
    vis_processor = registry.get_processor_class(
        vis_processor_cfg.name).from_config(vis_processor_cfg)

    # Initialize chat with the model and visual processor
    chat = Chat(model, vis_processor)

    print('Initialization Finished')

    return chat
