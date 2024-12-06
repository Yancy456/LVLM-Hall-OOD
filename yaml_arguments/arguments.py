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
