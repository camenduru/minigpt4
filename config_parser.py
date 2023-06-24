import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", type=str,
                        default='eval_configs/minigpt4.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.",
    )
    args = parser.parse_args()
    return args
