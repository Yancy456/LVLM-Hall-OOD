import argparse
import yaml

def yaml_bool(v):
    if v.lower() in ('True', 'true'):
        return True
    elif v.lower() in ('False', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Unsupported value {v}')


class ArgumentBase:
    def __init__(self):
        pass
    
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
        sys_args = self.sys_params()
        if sys_args.config == None:
            sys_args.config = self.default_config

        if cfg is None:
            with open(sys_args.config) as f:
                yaml_dict = yaml.safe_load(f)
                cfg = self.yaml_config(yaml_dict)