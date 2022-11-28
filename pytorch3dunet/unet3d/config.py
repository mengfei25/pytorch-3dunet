import argparse

import torch
import yaml

from pytorch3dunet.unet3d import utils

logger = utils.get_logger('ConfigLoader')


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    # for oob
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--precision', type=str, default='float32', help='precision')
    parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
    parser.add_argument('--num_iter', type=int, default=-1, help='num_iter')
    parser.add_argument('--num_warmup', type=int, default=-1, help='num_warmup')
    parser.add_argument('--profile', dest='profile', action='store_true', help='profile')
    parser.add_argument('--quantized_engine', type=str, default=None, help='quantized_engine')
    parser.add_argument('--ipex', dest='ipex', action='store_true', help='ipex')
    parser.add_argument('--jit', dest='jit', action='store_true', help='jit')

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warning('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    # for oob
    config['loaders']['batch_size'] = args.batch_size
    config['oob'] = {}
    config['oob']['precision'] = args.precision
    config['oob']['channels_last'] = args.channels_last
    config['oob']['num_iter'] = args.num_iter
    config['oob']['num_warmup'] = args.num_warmup
    config['oob']['profile'] = args.profile
    config['oob']['quantized_engine'] = args.quantized_engine
    config['oob']['ipex'] = args.ipex
    config['oob']['jit'] = args.jit

    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
