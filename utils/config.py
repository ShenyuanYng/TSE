import argparse
import yaml


def yaml_value(arg_str):
    """Parse scalar and collection CLI values without executing Python code."""
    return yaml.safe_load(arg_str)


def _val_to_argparse_kwargs(val):
    if isinstance(val, str):
        return {'type': strings_with_none, 'default': val}
    elif isinstance(val, bool):
        return {'type': bool_string_to_bool, 'default': val}
    else:
        return {'type': yaml_value, 'default': val}


def strings_with_none(arg_str):
    if arg_str.lower() in ['null', 'none']:
        return None
    else:
        return str(arg_str)


def bool_string_to_bool(bool_str):
    if str(bool_str).lower() == 'false':
        return False
    elif str(bool_str).lower() == 'true':
        return True
    else:
        raise


def yaml_to_parser(yaml_path):
    default_hyperparams = yaml.safe_load(open(yaml_path))
    parser = argparse.ArgumentParser()

    for k, v in default_hyperparams.items():
        argparse_kwargs = _val_to_argparse_kwargs(v)
        parser.add_argument('--{}'.format(k.replace('_', '-')),
                            **argparse_kwargs)
    return parser
