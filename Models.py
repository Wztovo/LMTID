from re import X
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils.rnn as rnn_utils
from torchvision.models import resnet50
from torchvision.models import densenet121


# 定义一个 Flatten 层，用于将高维张量展平成二维张量
class Flatten(nn.Module):
    def forward(self, input):
        # 将输入展平为 (batch_size, -1)
        return input.view(input.size(0), -1)

# 定义一个全连接块（包含 Flatten、Linear、ReLU、Dropout 层的组合）
class FcBlock(nn.Module):
    """
            初始化全连接块

            参数:
            - fc_params: 元组，表示全连接层的输入大小和输出大小，例如 (512, 256)。
            - flatten: 布尔值，是否需要在全连接层前加入 Flatten 层。
            """
    def __init__(self, fc_params, flatten):
        super(FcBlock, self).__init__()
        input_size = int(fc_params[0]) # 全连接层输入大小
        output_size = int(fc_params[1]) # 全连接层输出大小
        # 构建全连接层序列
        fc_layers = []
        if flatten:
            fc_layers.append(Flatten())  # 如果需要展平，则加入 Flatten 层
        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU()) # 全连接层
        fc_layers.append(nn.Dropout(0.5)) # Dropout，防止过拟合
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        # 执行前向传播
        fwd = self.layers(x)
        return fwd

# 定义一个卷积块（包含 Conv2D、BatchNorm、ReLU、AvgPool 层的组合）
class ConvBlock(nn.Module):
    def __init__(self, conv_params):
        """
                初始化卷积块

                参数:
                - conv_params: 元组，包含以下信息：
                    - input_channels: 输入通道数
                    - output_channels: 输出通道数
                    - avg_pool_size: 平均池化窗口大小
                    - batch_norm: 布尔值，是否启用批归一化
                """
        super(ConvBlock, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        avg_pool_size = conv_params[2]
        batch_norm = conv_params[3]
        # 构建卷积层序列
        conv_layers = []
        # 卷积层，使用 3x3 的卷积核，padding=1 保持特征图大小不变
        conv_layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))# 批归一化

        conv_layers.append(nn.ReLU())# 激活函数

        if avg_pool_size > 1:
            conv_layers.append(nn.AvgPool2d(kernel_size=avg_pool_size))# 平均池化层

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        # 执行前向传播
        fwd = self.layers(x)
        return fwd


class VGG(nn.Module):
    # 定义一个 VGG 网络结构
    def __init__(self, params):
        """
                初始化 VGG 网络

                参数:
                - params: 字典，包含以下配置信息：
                    - input_size: 输入图像的大小（通常为 32x32）
                    - num_classes: 分类任务的类别数量
                    - conv_channels: 每一层卷积的通道数列表
                    - fc_layers: 全连接层的大小列表
                    - max_pool_sizes: 每层卷积后的最大池化窗口大小
                    - conv_batch_norm: 是否启用批归一化
                    - init_weights: 是否初始化权重
                    - augment_training: 是否在训练时启用数据增强
                """
        super(VGG, self).__init__()

        self.input_size = int(params['input_size'])# 输入图像的大小
        self.num_classes = int(params['num_classes'])# 类别数量
        self.conv_channels = params['conv_channels']# 每层卷积的通道数
        self.fc_layer_sizes = params['fc_layers']# 全连接层的大小列表

        self.max_pool_sizes = params['max_pool_sizes']# 每层的最大池化窗口大小
        self.conv_batch_norm = params['conv_batch_norm'] # 是否启用批归一化
        self.init_weights = params['init_weights']# 是否初始化权重
        self.augment_training = params['augment_training']# 是否启用数据增强
        self.num_output = 1 # 默认输出为 1

        self.init_conv = nn.Sequential()# 可选的初始卷积层
        # 构建卷积块
        self.layers = nn.ModuleList()
        input_channel = 3 # 输入通道数（RGB 图像为 3 通道）
        cur_input_size = self.input_size # 当前的特征图大小
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size / 2)# 如果有最大池化，特征图大小减半
            conv_params = (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            self.layers.append(ConvBlock(conv_params))# 加入卷积块
            input_channel = channel # 更新下一层的输入通道数
        # 计算全连接层的输入大小
        fc_input_size = cur_input_size * cur_input_size * self.conv_channels[-1]
        # 构建全连接层
        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)# 全连接层的输入和输出大小
            flatten = False
            if layer_id == 0:
                flatten = True# 第一个全连接层需要展平输入

            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width # 更新下一层的输入大小
        # 构建最后的输出层
        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))# 最后一层全连接层
        end_layers.append(nn.Dropout(0.5))# Dropout 防止过拟合
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))# 输出分类层
        self.end_layers = nn.Sequential(*end_layers)
        # 如果需要初始化权重，则调用初始化函数
        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        fwd = self.init_conv(x)

        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class InvertedResidual(nn.Module):
    """
    MobileNetV2 的倒置残差模块，基于自定义的 ConvBlock。
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio, ConvBlock):
        """
        初始化倒置残差模块。

        参数：
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        - stride: 步幅
        - expand_ratio: 扩展倍率
        - ConvBlock: 自定义的 ConvBlock 类
        """
        super(InvertedResidual, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)  # 是否使用残差连接
        hidden_dim = in_channels * expand_ratio  # 扩展后的通道数

        layers = []
        # 如果扩展倍率不为 1，添加 Pointwise 卷积
        if expand_ratio != 1:
            conv_params = (in_channels, hidden_dim, 1, True)  # 输入通道，扩展通道，无池化，启用批归一化
            layers.append(ConvBlock(conv_params))  # Pointwise 卷积

        # 深度卷积（Depthwise Conv）
        depthwise_params = (hidden_dim, hidden_dim, 1, True)  # 通道数保持不变
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Pointwise 压缩卷积
        conv_params = (hidden_dim, out_channels, 1, True)  # 压缩到输出通道
        layers.append(ConvBlock(conv_params))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播。
        """
        if self.use_residual:
            return x + self.layers(x)  # 使用残差连接
        else:
            return self.layers(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 主网络结构。
    """
    def __init__(self, params, ConvBlock):
        """
        初始化 MobileNetV2。

        参数：
        - params: 配置参数字典，包括输入大小、类别数等。
        - ConvBlock: 自定义的 ConvBlock 类
        """
        super(MobileNetV2, self).__init__()

        self.input_size = int(params['input_size'])  # 输入图像大小
        self.num_classes = int(params['num_classes'])  # 分类类别数
        self.width_mult = params['width_mult']  # 宽度扩展倍率
        self.init_weights = params['init_weights']  # 是否初始化权重

        # MobileNetV2 主干配置，每个元组表示 (扩展倍率 t, 输出通道 c, 堆叠次数 n, 步幅 s)
        self.cfgs = [
            (1, 16, 1, 1),  # 第一组
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # 初始化第一层卷积层
        input_channel = int(32 * self.width_mult)  # 根据宽度倍率调整通道数
        self.last_channel = int(1280 * self.width_mult) if self.width_mult > 1.0 else 1280  # 最后一层通道数
        self.features = [ConvBlock((3, input_channel, 1, True))]  # 初始卷积层 (3 通道输入)

        # 构建主干网络
        for t, c, n, s in self.cfgs:
            output_channel = int(c * self.width_mult)  # 根据宽度倍率调整输出通道数
            for i in range(n):
                stride = s if i == 0 else 1  # 第一个模块使用传入步幅，后续步幅固定为 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t, ConvBlock))
                input_channel = output_channel

        # 添加最后的 Pointwise 卷积
        self.features.append(ConvBlock((input_channel, self.last_channel, 1, True)))
        self.features = nn.Sequential(*self.features)  # 将所有特征提取层组合为一个顺序模型

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Dropout 防止过拟合
            nn.Linear(self.last_channel, self.num_classes)  # 全连接层输出分类
        )

        # 初始化权重
        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        """
        前向传播。
        """
        x = self.features(x)  # 特征提取
        x = x.mean([2, 3])  # 全局平均池化 (batch_size, channels, height, width) -> (batch_size, channels)
        x = self.classifier(x)  # 分类层
        return x

    def initialize_weights(self):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class ResNet50(nn.Module):
    def __init__(self, params):
        """
        初始化 ResNet50 模型

        参数:
        - params: 包含模型配置信息的字典：
            - num_classes: 分类类别数
            - init_weights: 是否初始化权重
        """
        super(ResNet50, self).__init__()
        self.num_classes = params['num_classes']
        self.init_weights = params['init_weights']

        # 使用 torchvision 提供的预定义 ResNet50 模型
        self.model = resnet50(weights=None)
        # 修改全连接层的输出大小为指定的分类类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        # 初始化权重
        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class DenseNet121(nn.Module):
    def __init__(self, params):
        """
        初始化 DenseNet121 模型

        参数:
        - params: 字典，包含以下模型配置：
            - num_classes: 分类类别数
            - init_weights: 是否初始化权重
        """
        super(DenseNet121, self).__init__()
        self.num_classes = params['num_classes']
        self.init_weights = params['init_weights']

        # 使用 torchvision 提供的 DenseNet121 预定义模型
        self.model = densenet121(weights=None)  # 不加载预训练权重
        # 替换最后的全连接层（分类层）
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 增加 Dropout 比例，防止过拟合
            nn.Linear(self.model.classifier.in_features, self.num_classes)  # 全连接层输出分类
        )

        # 如果需要初始化权重，调用初始化函数
        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        """
        初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class TrData(Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]


def collate_fn(trs):
    onetr = trs[0]
    onepoint_size = onetr.size(1)
    input_size = onepoint_size - 1
    trs.sort(key=lambda x: len(x), reverse=True)
    tr_lengths = [len(sq) for sq in trs]
    trs = rnn_utils.pad_sequence(trs, batch_first=True, padding_value=0)
    var_x = trs[:, :, 1:input_size + 1]
    tmpy = trs[:, :, 0]
    var_y = tmpy[:, 0]
    return var_x, var_y, tr_lengths


class lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, num_layers=1, num_classes=2, batch_size=1):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, len_of_oneTr):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        batch_x_pack = rnn_utils.pack_padded_sequence(x,
                                                      len_of_oneTr, batch_first=True).cuda()
        out, (h1, c1) = self.layer1(batch_x_pack, (h0, c0))
        out = self.layer2(h1)
        return out, h1


class LSTM_Attention(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, num_layers=1, num_classes=2, batch_size=1):
        super(LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer3 = nn.Linear(hidden_size * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, len_of_oneTr):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        batch_x_pack = rnn_utils.pack_padded_sequence(x,
                                                      len_of_oneTr, batch_first=True).cuda()
        out, (h1, c1) = self.layer1(batch_x_pack, (h0, c0))
        outputs, lengths = rnn_utils.pad_packed_sequence(out, batch_first=True)
        permute_outputs = outputs.permute(1, 0, 2)
        atten_energies = torch.sum(h1 * permute_outputs,
                                   dim=2)

        atten_energies = atten_energies.t()

        scores = F.softmax(atten_energies, dim=1)

        scores = scores.unsqueeze(0)

        permute_permute_outputs = permute_outputs.permute(2, 1, 0)
        context_vector = torch.sum(scores * permute_permute_outputs,
                                   dim=2)
        context_vector = context_vector.t()
        context_vector = context_vector.unsqueeze(0)
        out2 = torch.cat((h1, context_vector), 2)
        out = self.layer3(out2)
        return out, out2


class CIFARData(Dataset):
    """
        自定义 PyTorch 数据集类，用于包装 CIFAR 数据。
        继承自 torch.utils.data.Dataset，需要实现 __len__ 和 __getitem__ 方法。
        """
    def __init__(self, X_train, y_train):
        """
               初始化数据集。

               参数:
               - X_train: 训练数据，通常为形状 (N, 3, 32, 32) 的 NumPy 数组或 Tensor，表示 N 张图像。
               - y_train: 标签数据，通常为形状 (N,) 的列表或数组，表示每张图像对应的分类标签。
               """
        self.X_train = X_train# 存储训练图像数据
        self.y_train = y_train# 存储训练标签

    def __len__(self):
        """
                返回数据集的大小。

                返回:
                - 数据集中样本的数量（即标签的数量）。
                """
        return len(self.y_train)# 标签的数量即为数据集的大小

    def __getitem__(self, idx):
        """
                根据索引获取数据集中的一组数据。

                参数:
                - idx: 数据索引，表示要获取第 idx 个样本。

                返回:
                - img: 第 idx 张图像，取自 X_train。
                - label: 第 idx 个标签，经过类型转换并转换为 PyTorch Tensor。
                """
        img = self.X_train[idx]# 根据索引获取对应的图像
        label = self.y_train[idx]# 根据索引获取对应的标签
        # 将标签转换为 NumPy 数组，并确保其数据类型为 int64
        label = np.array(label).astype(np.int64)
        # 将标签从 NumPy 数组转换为 PyTorch 的 Tensor
        label = torch.from_numpy(label)
        return img, label# 返回图像和对应的标签


class CIFARDataForDistill(Dataset):
    def __init__(self, X_train, y_train, softlabel):
        self.X_train = X_train
        self.y_train = y_train
        self.softlabel = softlabel

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        img = self.X_train[idx]
        label = self.y_train[idx]
        softlabel = self.softlabel[idx]
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label, softlabel
