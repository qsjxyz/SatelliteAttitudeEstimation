import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

import gc
import os
import time
import shutil
import importlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytorch3d import transforms

from lib.utils.warmup_scheduler import GradualWarmupScheduler
from lib.builder import Network, LossTrain
from lib.utils.register import import_all_modules_for_register, Registers
from lib.utils.testFunction import euler2RMSingleMode, quat2RMSingleMode, quat2RMMultiMode, \
    r6d2RMSingleMode, euler2RMMultiMode, manifold2RMSingleMode, manifold2RMMultiMode, r6d2RMMultiMode, \
    keypoint2RM, softEncodeQuat2RM, pointMultiModeR6d2RM

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='E122_RGB_R6d_PointFusion_NumPoint512')
opt = parser.parse_args()
config = importlib.import_module('config.'+opt.config).cfg

def test(cfg):
    checkpointPath = os.path.join(cfg['output_path'], cfg['name'])

    cuda_gpu = torch.cuda.is_available()  # 判断GPU是否存在可用
    if not cuda_gpu:
        raise EnvironmentError("无法正常使用GPU")
    print('gpu:', cuda_gpu)
    torch.backends.cudnn.enabled = False

    # 加载训练集train与评估集val
    test = Registers.DATASETS[cfg['dataset']['type']] \
        ('test',
         cfg['dataset']['img_size'],
         cfg['dataset']['dataset_path'])

    # 设置batchsize
    loader = DataLoader(test, batch_size=cfg['test']['batch_size'], shuffle=True)

    net = Network(cfg)
    net = torch.nn.DataParallel(net).cuda()

    net.load_state_dict(torch.load(os.path.join(checkpointPath, cfg['test']['resume'])), strict=True)
    result = {"error_alpha": [], "error_beta": [], "error_gamma": [], "euler_angle_error": [],
              "er": [], "add": [], "cls": [], "confidence": []}

    totalTime = 0.

    with torch.no_grad():
        net.eval()
        # 训练，遍历所有图像作为一个epoch
        for step, data_batch in tqdm(enumerate(loader)):
            cls = data_batch['class']

            input_network = [data_batch[name].cuda() for name in cfg['backbone']['input_for_forward']]
            startTime = time.time()
            predict = net(input_network)  # 计算预测值

            if cfg['head']['output_parameter_num'] == 1:
                predict = [predict]

            input = [predict[index].cuda() for index in cfg['test']['pose_data_id']]
            if cfg['test']['extra_parameter'] is None:
                extra_parameter = None
            else:
                extra_parameter = data_batch[cfg['test']['extra_parameter']]

            if cfg['test']['pose_type'] == 'singleEuler':
                rm_pred = euler2RMSingleMode(input, extra_parameter)
            elif cfg['test']['pose_type'] == 'singleQuat':
                rm_pred = quat2RMSingleMode(input, extra_parameter)
            elif cfg['test']['pose_type'] == 'singleR6d':
                rm_pred = r6d2RMSingleMode(input, extra_parameter)
            elif cfg['test']['pose_type'] == 'singleManifold':
                rm_pred = manifold2RMSingleMode(input, extra_parameter)
            elif cfg['test']['pose_type'] == 'multiEuler':
                rm_pred = euler2RMMultiMode(input, extra_parameter)
            elif cfg['test']['pose_type'] == 'multiQuat':
                rm_pred = quat2RMMultiMode(input, extra_parameter)
            elif cfg['test']['pose_type'] == 'multiR6d':
                rm_pred = r6d2RMMultiMode(input, extra_parameter)
            elif cfg['test']['pose_type'] == 'multiManifold':
                rm_pred = manifold2RMMultiMode(input, extra_parameter)
            elif cfg['test']['pose_type'] == 'keypoint':
                rm_pred = keypoint2RM(input, extra_parameter)
            elif cfg['test']['pose_type'] == 'softEncodeQuat':
                rm_pred = softEncodeQuat2RM(input, extra_parameter)
            elif cfg['test']['pose_type'] == 'pointMultiMode':
                rm_pred = pointMultiModeR6d2RM(input, extra_parameter)

            endTime = time.time()
            totalTime += endTime - startTime

            euler_truth = data_batch['meta'].cuda() / np.pi * 180.
            rm_truth = data_batch['RM'].cuda()
            quat_truth = data_batch['quat'].cuda()

            rm_pred, confidence = rm_pred[0], rm_pred[1]
            euler_pred = transforms.matrix_to_euler_angles(rm_pred, "XYZ") / np.pi * 180.
            rm_pred = rm_pred
            quat_pred = transforms.matrix_to_quaternion(rm_pred)

            error_alpha = torch.min(torch.abs(euler_pred[:, 0] - euler_truth[:, 0]), 360 - torch.abs(euler_pred[:, 0] - euler_truth[:, 0]))
            error_beta = torch.abs(euler_pred[:, 1] - euler_truth[:, 1])
            error_gamma = torch.min(torch.abs(euler_pred[:, 2] - euler_truth[:, 2]), 360 - torch.abs(euler_pred[:, 2] - euler_truth[:, 2]))
            euler_angle_error = torch.sqrt((error_gamma ** 2 + error_beta ** 2 + error_alpha ** 2) / 3)

            pred_acos = transforms.standardize_quaternion(quat_pred).float()
            gt_acos = transforms.standardize_quaternion(quat_truth).float()
            er = 2 * torch.acos(F.cosine_similarity(pred_acos, gt_acos, dim=-1))

            point = data_batch['ori_model'].cuda().unsqueeze(2)
            rm_truth = rm_truth.unsqueeze(1)
            rm_pred = rm_pred.unsqueeze(1)
            point_truth = torch.matmul(point, rm_truth).squeeze()
            point_pred = torch.matmul(point, rm_pred).squeeze()
            add = torch.mean(torch.norm((point_pred - point_truth), dim=2), dim=1)

            result['cls'] += cls
            result['error_alpha'] += list(error_alpha.cpu().numpy())
            result['error_beta'] += list(error_beta.cpu().numpy())
            result['error_gamma'] += list(error_gamma.cpu().numpy())
            result['euler_angle_error'] += list(euler_angle_error.cpu().numpy())
            result['er'] += list(er.cpu().numpy())
            result['add'] += list(add.cpu().numpy())
            result['confidence'] += list(torch.exp(confidence).cpu().numpy())
            # 释放不用的内存，防止程序越跑越慢
            gc.collect()
            torch.cuda.empty_cache()
    dataTable = pd.DataFrame(result)
    dataTable.to_excel(os.path.join(checkpointPath, 'debug.xlsx'))
    dataTable = dataTable[dataTable['confidence'] <= 0.147]

    outDict = {}

    objlist = test.objlist
    # 计算每一个类别的样本数量
    objNum = {'all': 0.}
    for object in objlist:
        objNum[object] = len(dataTable[dataTable['cls'] == object])
        objNum['all'] += objNum[object]

    # 欧拉角相关评估指标
    outDict['yawMean'] = dataTable['error_alpha'].mean()
    outDict['pitchMean'] = dataTable['error_beta'].mean()
    outDict['rollMean'] = dataTable['error_gamma'].mean()
    outDict['yawMid'] = dataTable['error_alpha'].median()
    outDict['pitchMid'] = dataTable['error_beta'].median()
    outDict['rollMid'] = dataTable['error_gamma'].median()
    outDict['yawL5'] = len(dataTable[dataTable['error_alpha'] < 5]) / objNum['all']
    outDict['pitchL5'] = len(dataTable[dataTable['error_beta'] < 5]) / objNum['all']
    outDict['rollL5'] = len(dataTable[dataTable['error_gamma'] < 5]) / objNum['all']
    outDict['eulerMean'] = dataTable['euler_angle_error'].mean()
    outDict['eulerStd'] = dataTable['euler_angle_error'].std()
    outDict['eulerMid'] = dataTable['euler_angle_error'].median()
    outDict['eulerL10'] = len(dataTable[dataTable['euler_angle_error'] < 10]) / objNum['all']
    for object in objlist:
        outDict[object + 'EulerMean'] = dataTable[dataTable['cls'] == object]['euler_angle_error'].mean()
        outDict[object + 'EulerMid'] = dataTable[dataTable['cls'] == object]['euler_angle_error'].median()

    # 重投影误差相关指标
    outDict['ADDMean'] = dataTable['add'].mean()
    outDict['ADDStd'] = dataTable['add'].std()
    outDict['ADDMid'] = dataTable['add'].median()
    outDict['ADD10'] = len(dataTable[dataTable['add'] < 0.2]) / objNum['all']
    outDict['ADD2'] = len(dataTable[dataTable['add'] < 0.04]) / objNum['all']
    for object in objlist:
        outDict[object + 'ADD10'] = len(dataTable[(dataTable['cls']==object)&(dataTable['add']<0.2)]) / objNum[object]
        outDict[object + 'ADD2'] = len(dataTable[(dataTable['cls']==object)&(dataTable['add']<0.04)]) / objNum[object]

    outDict['erMean'] = dataTable['er'].mean()
    outDict['erStd'] = dataTable['er'].std()
    outDict['erMid'] = dataTable['er'].median()
    outDict['speed'] = totalTime / objNum['all']

    outDataTable = pd.DataFrame.from_dict(outDict,orient='index').T
    outDataTable.to_excel(os.path.join(checkpointPath, 'result.xlsx'), index=False)

if __name__ == '__main__':
    import_all_modules_for_register(config)
    test(config)
