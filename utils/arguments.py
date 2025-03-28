import argparse
import yaml
from yaml_arguments.argument_base import yaml_bool

cfg = None  # global sharing configurations


def image_shape(s: str):
    shape = [int(x) for x in s.split(',')]
    return shape


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
        parser = argparse.ArgumentParser()

        # Model configs and path
        parser.add_argument("--model_name", required=True, type=str)
        parser.add_argument(
            "--model_path", required=True, type=str)
        parser.add_argument(
            "--image_shape", required=False, default=None, type=image_shape)
        parser.add_argument(
            "--n_multi_gene", required=False, default=0, type=int)
        parser.add_argument(
            "--return_logits", required=False, default=False, type=yaml_bool)

        # Dataset configurations
        parser.add_argument("--dataset", required=True, type=str)
        parser.add_argument("--data_folder", required=False,
                            type=str, default=None)
        parser.add_argument("--annotation_path",
                            required=False, type=str, default=None)

        parser.add_argument("--split", default="val", type=str)
        parser.add_argument("--category", type=str, default=None)
        parser.add_argument(
            "--save_path", required=True, type=str)
        parser.add_argument(
            "--batch_size", default=1, type=int)
        parser.add_argument("--num_samples", type=int,
                            help='the number of samples')
        parser.add_argument(
            "--shuffle", type=yaml_bool, default=False, help='whether shuffle data')

        # LLM generation configurations
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--num_beams", type=int, default=1)
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
