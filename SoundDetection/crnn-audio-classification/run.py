#!/usr/bin/env python3
import argparse, json, os
import torch

from utils import Logger
import data as data_module
import net as net_module
from train import Trainer
from eval import ClassificationEvaluator, AudioInference

# Helper: 获取数据增强变换对象
def _get_transform(config, name):
    tsf_name = config['transforms']['type']  # 读取配置中变换类型
    tsf_args = config['transforms']['args']  # 变换的参数
    return getattr(data_module, tsf_name)(name, tsf_args)

# Helper: 获取模型的基本信息
def _get_model_att(checkpoint):
    m_name = checkpoint['config']['model']['type']  # 模型名称
    sd = checkpoint['state_dict']                  # 模型的状态字典
    classes = checkpoint['classes']                # 模型的分类类别
    return m_name, sd, classes

# 模型评估主函数
def eval_main(checkpoint):
    config = checkpoint['config']  # 加载配置
    data_config = config['data']   # 数据配置

    # 获取数据增强变换和测试数据加载器
    tsf = _get_transform(config, 'val')
    data_manager = getattr(data_module, config['data']['type'])(config['data'])
    test_loader = data_manager.get_loader('val', tsf)

    # 加载模型
    m_name, sd, classes = _get_model_att(checkpoint)
    model = getattr(net_module, m_name)(classes, config, state_dict=sd)
    print(model)

    # 加载模型权重
    model.load_state_dict(checkpoint['state_dict'])

    # 获取评估指标
    num_classes = len(classes)
    metrics = getattr(net_module, config['metrics'])(num_classes)

    # 运行评估
    evaluation = ClassificationEvaluator(test_loader, model)
    ret = evaluation.evaluate(metrics)
    print(ret)
    return ret

# 单文件推理主函数
def infer_main(file_path, config, checkpoint):
    # 如果没有检查点，初始化模型
    if checkpoint is None:
        model = getattr(net_module, config['model']['type'])()
    else:
        m_name, sd, classes = _get_model_att(checkpoint)
        model = getattr(net_module, m_name)(classes, config, state_dict=sd)
        model.load_state_dict(checkpoint['state_dict'])

    # 数据增强变换
    tsf = _get_transform(config, 'val')

    # 创建推理器，执行推理并绘图
    inference = AudioInference(model, transforms=tsf)
    label, conf = inference.infer(file_path)
    print(label, conf)

    # 绘图
    inference.draw(file_path, label, conf)

# Helper: 筛选可训练参数
def requires_grad_filter(p):
    return p.requires_grad

# 训练主函数
def train_main(config, resume):
    train_logger = Logger()  # 日志管理器

    data_config = config['data']  # 数据配置

    # 加载训练和验证数据增强变换
    t_transforms = _get_transform(config, 'train')
    v_transforms = _get_transform(config, 'val')
    print(t_transforms)

    # 加载数据加载器和分类类别
    data_manager = getattr(data_module, config['data']['type'])(config['data'])
    classes = data_manager.classes
    t_loader = data_manager.get_loader('train', t_transforms)
    v_loader = data_manager.get_loader('val', v_transforms)

    # 加载模型
    m_name = config['model']['type']
    model = getattr(net_module, m_name)(classes, config=config)
    num_classes = len(classes)

    # 加载损失函数和评估指标
    loss = getattr(net_module, config['train']['loss'])
    metrics = getattr(net_module, config['metrics'])(num_classes)

    # 筛选模型的可训练参数
    trainable_params = filter(requires_grad_filter, model.parameters())

    # 加载优化器
    opt_name = config['optimizer']['type']
    opt_args = config['optimizer']['args']
    optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)

    # 加载学习率调度器
    lr_name = config['lr_scheduler']['type']
    lr_args = config['lr_scheduler']['args']
    if lr_name == 'None':
        lr_scheduler = None
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)

    # 创建 Trainer，执行训练
    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=t_loader,
                      valid_data_loader=v_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()
    return trainer

# 测试数据加载器
def _test_loader(config):
    # Helper: 显示一个批次的数据
    def disp_batch(batch):
        ret = []
        for b in batch:
            if len(b.size()) != 1:
                ret.append(b.shape)
            else:
                ret.append(b)
        return ret

    # 数据增强变换和加载器
    tsf = _get_transform(config, 'train')
    data_manager = getattr(data_module, config['data']['type'])(config['data'])
    loader = data_manager.get_loader('train', tsf)
    print(tsf.transfs)

    for batch in loader:
        print(disp_batch([batch[0], batch[-1]]))

# 程序入口
if __name__ == '__main__':
    # 命令行参数解析
    argparser = argparse.ArgumentParser(description='PyTorch Template')

    # action：必须参数，指定用户要执行的操作（训练、测试或评估）。
    # --config：可选参数，指定配置文件路径，供后续使用。
    # --resume：可选参数，指定恢复模型的检查点文件路径。
    # --net_mode：可选参数，指定迁移学习的类型，默认为init。
    # --cfg：可选参数，用于指定神经网络层的配置文件。
    argparser.add_argument('action', type=str, help='what action to take (train, test, eval)')
    argparser.add_argument('-c', '--config', default=None, type=str, help='config file path')
    argparser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint')
    argparser.add_argument('--net_mode', default='init', type=str, help='type of transfer learning to use')
    argparser.add_argument('--cfg', default=None, type=str, help='nn layer config file')

    args = argparser.parse_args()   #解析命令行传入的参数，并将其存储在args对象中

    # 配置加载：优先加载 config，如果无 config 则加载检查点
    # 加载配置或检查点
    checkpoint = None
    if args.config:
        config = json.load(open(args.config))
        config['net_mode'] = args.net_mode
        config['cfg'] = args.cfg
    elif args.resume:
        # 修改这里，将模型加载到CPU
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        config = checkpoint['config']
    else:
        raise AssertionError("Configuration file needs to be specified.")

    # 根据命令行参数选择模式
    if args.action == 'train':
        train_main(config, args.resume)
    elif args.action == 'eval':
        eval_main(checkpoint)
    elif args.action == 'testloader':
        _test_loader(config)
    elif os.path.isfile(args.action):  # 如果传入的是文件路径，执行推理
        file_path = args.action
        infer_main(file_path, config, checkpoint)
