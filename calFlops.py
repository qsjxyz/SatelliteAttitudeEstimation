import torch

from ptflops import get_model_complexity_info
import importlib
import argparse

from lib.utils.warmup_scheduler import GradualWarmupScheduler
from lib.builder import Network, LossTrain, LossVal
from lib.utils.register import import_all_modules_for_register, Registers

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='E008_RGB_EulerLog_R50_wobottle')
opt = parser.parse_args()
config = importlib.import_module('config.'+opt.config).cfg

def calFlops(cfg):
    net = Network(cfg)
    macs, params = get_model_complexity_info(net,
                                             (1, cfg['backbone']['input_channel'], cfg['dataset']['img_size'], cfg['dataset']['img_size']),
                                             as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':
    import_all_modules_for_register(config)
    calFlops(config)
