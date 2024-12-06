import argparse
import yaml

cfg = None  # global sharing configurations


class Arguments:
    def __init__(self, default_config=None) -> None:
        self.default_config = default_config
        self.load_config()

    def sys_params(self):
        '''load params from command line'''
        sys_parser = argparse.ArgumentParser()
        sys_parser.add_argument('--config', type=str,
                                help='the path of configuration yaml')
        return sys_parser.parse_args()

    def yaml_config(self, dict):
        '''where define the parameters in yaml configuration files'''
        parser = argparse.ArgumentParser(
            description='Process yaml parameters.')

        parser.add_argument("--model_name", default="LLaVA-7B")
        parser.add_argument(
            "--model_path", default="/root/autodl-tmp/liuhaotian/llava-7b")
        parser.add_argument(
            "--save_path", default="./output/save_for_eval_linear")
        parser.add_argument("--num_samples", type=int,
                            default=None, help='the number of samples')
        parser.add_argument(
            "--sampling", choices=['first', 'random', 'class'], default='random')
        parser.add_argument("--split", default="val")
        parser.add_argument("--dataset", default="MAD")
        parser.add_argument("--prompt", default='mq')
        parser.add_argument("--theme", default='mad')
        parser.add_argument("--answers_file", type=str,
                            default="./output/tmp/0_0.jsonl")
        parser.add_argument("--num_chunks", type=int, default=1)
        parser.add_argument("--chunk_idx", type=int, default=0)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--num_beams", type=int, default=1)
        parser.add_argument("--judge_path", type=str)
        parser.add_argument("--token_id", type=int, default=0,
                            help='the index of token that is used to classification. 0 means the first token.')

        return self.load_dict_params(dict, parser)

    def load_dict_params(self, dict, parser):
        def parse_params_from_dict(dict):
            args = []
            for key, value in dict.items():
                args.append(f'--{key}={value}')
            return parser.parse_args(args)

        # Parse the parameters from the dictionary
        args = parse_params_from_dict(dict)
        return args

    def load_config(self):
        global cfg
        sys_args = self.sys_params()
        if sys_args.config == None:
            sys_args.config = self.default_config

        if cfg is None:
            with open(sys_args.config) as f:
                yaml_dict = yaml.safe_load(f)
                cfg = self.yaml_config(yaml_dict)

    def get_config(self):
        return cfg
