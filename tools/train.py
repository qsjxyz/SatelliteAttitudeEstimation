import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import gc
import os
import shutil
import importlib
import argparse
import numpy as np
from tqdm import tqdm

from lib.utils.warmup_scheduler import GradualWarmupScheduler
from lib.builder import Network, LossTrain, LossVal
from lib.utils.register import import_all_modules_for_register, Registers

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='E019_RGB_MultiR6dLog')
opt = parser.parse_args()
config = importlib.import_module('config.'+opt.config).cfg

def train(cfg):
    # tensorboard，训练信息存储位置
    summaryPath = os.path.join(cfg['output_path'], cfg['name'], 'log')
    writer = SummaryWriter(summaryPath)

    configOriPath = os.path.join('../config', opt.config+'.py')
    configOutPath = os.path.join(cfg['output_path'], cfg['name'], 'config.py')
    shutil.copy(configOriPath, configOutPath)

    checkpointPath = os.path.join(cfg['output_path'], cfg['name'])

    cuda_gpu = torch.cuda.is_available()  # 判断GPU是否存在可用
    if not cuda_gpu:
        raise EnvironmentError("无法正常使用GPU")
    print('gpu:', cuda_gpu)
    torch.backends.cudnn.enabled = False

    # 加载训练集train与评估集val
    train = Registers.DATASETS[cfg['dataset']['type']] \
        ('train',
         cfg['dataset']['img_size'],
         cfg['dataset']['dataset_path'])
    val = Registers.DATASETS[cfg['dataset']['type']] \
        ('val',
         cfg['dataset']['img_size'],
         cfg['dataset']['dataset_path'])

    # 设置batchsize
    loader = DataLoader(train, batch_size=cfg['train']['batch_size'], shuffle=True)
    loader_val = DataLoader(val, batch_size=cfg['train']['batch_size'], shuffle=True)

    net = Network(cfg)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    net = torch.nn.DataParallel(net).cuda()

    if not cfg['train']['resume'] is None:
        net.load_state_dict(torch.load(os.path.join(checkpointPath, cfg['train']['resume'])), strict=True)

    # 损失函数
    loss_train_func = LossTrain(cfg)
    loss_train_func.cuda()
    loss_val_func = LossVal(cfg)
    loss_val_func.cuda()

    # 定义warmup轮数total_epoch、到达最大值后学习率下降的方式
    scheduler_steplr = StepLR(optimizer, step_size=cfg['train']['steplr'], gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg['train']['warmup'],
                                              after_scheduler=scheduler_steplr)

    val_result = []
    total_step = 0
    for epoch in range(cfg['train']['start_epoch']+1, cfg['train']['epoch']+1):
        # train
        net.train()
        # 训练，遍历所有图像作为一个epoch
        for step, data_batch in enumerate(loader):
            total_step += 1

            input_network = [data_batch[name].cuda() for name in cfg['backbone']['input_for_forward']]
            predict = net(input_network)  # 计算预测值
            if cfg['head']['output_parameter_num'] == 1:  # 判断转化为列表后，是列表有多个输出参数还是生成了一个输出参数的列表化
                predict = [predict]

            input_loss = list(predict) + [data_batch[name].cuda() for name in cfg['lossTrain']['input_for_forward']]
            loss = loss_train_func(input_loss)

            optimizer.zero_grad()  # 清除累计梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 优化

            scheduler_warmup.step(epoch)  # 获取当前学习率

            if step % 5 == 0:  # 每训练5步(每一步中训练batch_size张图片)显示一次当前训练情况
                print('epoch:{}, step:{}, lr:{}, loss:{}'.format(epoch, step, scheduler_warmup.get_lr()[0], loss))
            if total_step % 50 == 0:
                writer.add_scalar('loss', loss, total_step)

        # validate
        if not (epoch % cfg['train']['val_frequency'] == 0):  # 调整validate的次数
            continue
        net.eval()
        print('epoch:{}, validation...'.format(epoch))

        # 存储结果
        valLossList = []
        with torch.no_grad():
            for data_batch in tqdm(loader_val):
                input_network = [data_batch[name].cuda() for name in cfg['backbone']['input_for_forward']]
                predict = net(input_network)  # 计算预测值

                if cfg['head']['output_parameter_num'] == 1:
                    predict = [predict]
                input_loss = [predict[index].cuda() for index in cfg['lossVal']['output_for_forward']] + \
                             [data_batch[name].cuda() for name in cfg['lossVal']['input_for_forward']]
                loss = loss_val_func(input_loss)

                valLossList.append(loss.cpu().detach().numpy())

        loss = np.array(valLossList).mean()
        print('loss:{}'.format(loss))
        val_result.append(loss)
        writer.add_scalar('val_loss', loss, epoch)

        torch.save(net.state_dict(), os.path.join(checkpointPath, 'model'+str(epoch).zfill(3)+'.pkl'))

        # 释放不用的内存，防止程序越跑越慢
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(net.state_dict(), os.path.join(checkpointPath, 'model_last.pkl'))
    writer.close()

if __name__ == '__main__':
    import_all_modules_for_register()
    train(config)
