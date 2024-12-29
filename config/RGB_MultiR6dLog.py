cfg = {'name': 'RGB_MultiR6dLog',
       'dataset':
           {'type': 'BUAASID',
            'img_size': 224,
            'dataset_path': '/home/d409/SatellitePoseUni6D/data/'
            },
       'backbone':
           {'type': 'resnet18',
            'input_channel': 3,
            'output_channel': 512,
            'input_for_forward': ['img']
            },
       'head':
           {'type': 'multiR6dHead',
            'input_channel': 512,
            'extra_parameter': {'nm': 8},
            'output_parameter_num': 2
            },
       'lossTrain':
           {'type': 'multiR6dLoss',
            'extra_parameter': {'func': 'LOG', 'nm': 8},
            'input_for_forward': ['r6d']
            },
       'lossVal':
           {'type': 'addLossMultiR6dMode',
            'extra_parameter': {'mn': 8},
            'input_for_forward': ['r6d', 'point'],
            'output_for_forward': [0, 1]
            },
       'train':
           {'baselr': 0.1,
            'warmup': 10,
            'steplr': 40,
            'batch_size': 128,
            'epoch': 150,
            'val_frequency': 5,
            'resume': None,
            'start_epoch': 0
            },
       'test':
           {'model_point_num': 1024,
            'batch_size': 128,
            'resume': 'model_last.pkl',
            'pose_type': 'multiR6d',
            'extra_parameter': None,
            'pose_data_id': [0, 1],  # 如果是多输出，需要返回有用输出的id
            },
       'output_path': '/home/d409/SatellitePoseEstimation/result'
       }