import pickle
from tkinter import Y
import numpy as np
import os
from PIL import Image
import scipy.io as sio
import pandas as pd

def readCIFAR10(data_path):
    """
      读取CIFAR-10数据集的训练集和测试集数据，并返回图像数据和对应的标签。

      CIFAR-10数据集由5个训练批次（data_batch_1到data_batch_5）和1个测试批次（test_batch）组成。
      每个批次包含10000张32x32的彩色图片，以及对应的类别标签（10个分类）。

      参数：
      - data_path (str): CIFAR-10数据集的根目录路径。

      返回：
      - X (numpy.ndarray): 训练集的图像数据，形状为 (50000, 3072)，
                           其中3072表示32x32x3（行、列、通道）的展开形式。
      - y (numpy.ndarray): 训练集的标签数据，形状为 (50000,)，
                           每个标签表示图像的类别（范围为0到9）。
      - XTest (numpy.ndarray): 测试集的图像数据，形状为 (10000, 3072)，
                               格式同训练集。
      - yTest (numpy.ndarray): 测试集的标签数据，形状为 (10000,)，
                               格式同训练集。

      文件说明：
      - data_batch_1到data_batch_5：训练集数据，每个文件包含10000条记录。
      - test_batch：测试集数据，包含10000条记录。
      - 数据存储在字典格式中，键值如下：
        - "data"：一个二维数组，形状为 (10000, 3072)，每行是一个图像的像素值。
        - "labels"：一个列表，包含每个图像对应的类别标签。

      示例用法：
      ```
      train_images, train_labels, test_images, test_labels = readCIFAR10('./cifar-10-batches-py')
      print(train_images.shape)  # (50000, 3072)
      print(train_labels.shape)  # (50000,)
      print(test_images.shape)   # (10000, 3072)
      print(test_labels.shape)   # (10000,)
      ```
      """
    print(data_path)# 打印提供的数据路径，便于调试和检查输入路径是否正确
    # 遍历5个训练批次文件
    for i in range(5):
        # 打开第 i+1 个训练批次文件
        f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
        # 使用 pickle 加载数据，指定编码格式为 'iso-8859-1'，以保证兼容性
        train_data_dict = pickle.load(f, encoding='iso-8859-1')
        f.close()# 关闭文件以释放资源
        # 如果是第一个批次，初始化训练集数据和标签
        if i == 0:
            X = train_data_dict["data"]# 提取图像数据（形状为 (10000, 3072)）
            y = train_data_dict["labels"]# 提取标签数据（长度为 10000）
            continue# 跳过后续代码，直接进入下一次循环
        # 将当前批次的图像数据和标签数据追加到已有数据中
        # np.concatenate 用于在第0维（样本数量）上拼接数组
        X = np.concatenate((X, train_data_dict["data"]), axis=0)
        y = np.concatenate((y, train_data_dict["labels"]), axis=0)
    # 加载测试批次文件
    f = open(data_path + '/test_batch', 'rb')
    # 使用 pickle 加载数据，格式与训练批次相同
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()# 关闭文件
    # 提取测试集的图像数据和标签数据
    XTest = np.array(test_data_dict["data"])
    yTest = np.array(test_data_dict["labels"])
    # 返回训练集和测试集的数据与标签
    return X, y, XTest, yTest

def readCIFAR100(data_path):
    print(data_path)
    f = open(data_path + '/train' , 'rb')
    train_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()
    X = np.array(train_data_dict["data"])  # 训练集图像数据，形状为 (50000, 3072)
    y = np.array(train_data_dict["fine_labels"])  # 训练集标签数据，形状为 (50000,)
    f = open(data_path + '/test', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()
    XTest = np.array(test_data_dict["data"])
    yTest = np.array(test_data_dict["fine_labels"])

    return X, y, XTest, yTest
def readCINIC10(data_path):
    """
    读取 CINIC-10 数据集，将训练集和测试集的图像数据展平为形状 (N, 3072)，并返回对应的标签。

    参数：
    - data_path (str): CINIC-10 数据集的根目录路径。

    返回：
    - X_train (numpy.ndarray): 训练集的图像数据，形状为 (N_train, 3072)。
    - y_train (numpy.ndarray): 训练集的标签数据，形状为 (N_train,)。
    - X_test (numpy.ndarray): 测试集的图像数据（验证集和测试集合并），形状为 (N_test, 3072)。
    - y_test (numpy.ndarray): 测试集的标签数据，形状为 (N_test,)。
    """
    def load_split(split):
        images = []
        labels = []
        classes = sorted(os.listdir(os.path.join(data_path, split)))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_path = os.path.join(data_path, split, cls_name)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = np.array(img).astype(np.float32) / 255.0
                    images.append(img.flatten())
                    labels.append(class_to_idx[cls_name])

        return np.array(images), np.array(labels)

    X_train, y_train = load_split('train')
    X_valid, y_valid = load_split('valid')
    X_test, y_test = load_split('test')

    # 合并验证集和测试集
    X_test = np.concatenate((X_valid, X_test), axis=0)
    y_test = np.concatenate((y_valid, y_test), axis=0)

    return X_train, y_train, X_test, y_test

def readSVHN(data_path):
    """
    读取SVHN数据集的训练集和测试集数据，并返回图像数据和对应的标签。

    SVHN数据集由以下文件组成：
    - train_32x32.mat：训练集
    - test_32x32.mat：测试集

    参数：
    - data_path (str): SVHN数据集的根目录路径。

    返回：
    - X (numpy.ndarray): 训练集的图像数据，形状为 (N_train, 3072)，
                         其中3072表示32x32x3（行、列、通道）的展开形式。
    - y (numpy.ndarray): 训练集的标签数据，形状为 (N_train,)，
                         每个标签表示图像的类别（范围为0到9）。
    - XTest (numpy.ndarray): 测试集的图像数据，形状为 (N_test, 3072)，
                             格式同训练集。
    - yTest (numpy.ndarray): 测试集的标签数据，形状为 (N_test,)，
                             格式同训练集。
    """
    # 加载训练集
    train_data = sio.loadmat(f"{data_path}/train_32x32.mat")
    X_train = train_data['X']  # 图像数据，形状为 (32, 32, 3, N_train)
    y_train = train_data['y']  # 标签数据，形状为 (N_train, 1)

    # 加载测试集
    test_data = sio.loadmat(f"{data_path}/test_32x32.mat")
    X_test = test_data['X']  # 图像数据，形状为 (32, 32, 3, N_test)
    y_test = test_data['y']  # 标签数据，形状为 (N_test, 1)

    # 转换图像数据格式，从 (32, 32, 3, N) 转换为 (N, 32, 32, 3)
    X_train = np.transpose(X_train, (3, 0, 1, 2))
    X_test = np.transpose(X_test, (3, 0, 1, 2))

    # 展平图像数据，从 (N, 32, 32, 3) 转换为 (N, 3072)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # 将标签从 (N, 1) 转换为 (N,) 并将 "10" 替换为 "0"
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    return X_train, y_train, X_test, y_test
def reshape_for_save(raw_data):
    """
        将 CIFAR 数据重新整形为适合保存或进一步处理的格式。

        参数:
        - raw_data: 原始数据，形状为 (N, 3072)，N 为样本数量。
          每个样本由 3072 个特征值表示，其中前 1024 个是红色通道值，
          中间 1024 个是绿色通道值，最后 1024 个是蓝色通道值。

        返回:
        - reshaped_data: 重新整形后的数据，形状为 (N, 3, 32, 32)，
          其中每个样本被重新排列为 (通道数, 高度, 宽度) 格式的图像数据。
        """
    # 将原始数据分割成 RGB 三个通道，分别是 (N, 1024)，然后沿最后一个维度堆叠，得到 (N, 1024, 3)
    raw_data = np.dstack((raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
    # 将数据重新整形为 (N, 32, 32, 3)，即将每个样本的 1024 个像素值映射到 (高 32, 宽 32, 通道数 3)
    raw_data = raw_data.reshape((raw_data.shape[0], 32, 32, 3))
    # 调整数据的轴顺序为 (N, 通道数, 高度, 宽度)，从 (N, 32, 32, 3) 转换为 (N, 3, 32, 32)
    raw_data = raw_data.transpose(0, 3, 1, 2)
    # 转换数据类型为 float32，以适应后续处理（例如标准化）
    return raw_data.astype(np.float32)


def rescale(raw_data, offset, scale):
    """
       对原始数据进行标准化处理，将数据的均值（offset）调整为 0，标准差（scale）调整为 1。

       参数:
       - raw_data: 原始数据，形状为 (N, 3072)，N 为样本数量。
       - offset: 每个特征的均值，形状为 (3, 32, 32)。
       - scale: 每个特征的标准差，形状为 (3, 32, 32)。

       返回:
       - normalized_data: 标准化后的数据，形状为 (N, 3, 32, 32)，每个样本的特征值被归一化。
       """
    # 首先调用 reshape_for_save 将原始数据转换为 (N, 3, 32, 32) 的形状
    newdata = reshape_for_save(raw_data)
    # 标准化处理：(数据 - 均值) / 标准差，保证数据均值为 0，标准差为 1
    return (newdata - offset) / scale


def preprocessingCIFAR(toTrainData, toTestData):
    """
       对 CIFAR 数据集的训练数据和测试数据进行标准化预处理。
       标准化包括计算均值（offset）和标准差（scale），
       并将数据进行归一化处理。

       参数:
       - toTrainData: 训练数据，形状为 (N_train, D)，N_train 为样本数量，D 为特征维度。
       - toTestData: 测试数据，形状为 (N_test, D)，N_test 为样本数量，D 为特征维度。如果为空，则仅处理训练数据。

       返回:
       - 如果 toTestData 不为空:
           - 标准化后的训练数据 (numpy.ndarray)
           - 标准化后的测试数据 (numpy.ndarray)
       - 如果 toTestData 为空:
           - 标准化后的训练数据 (numpy.ndarray)
       """
    # 检查测试数据是否为空
    if (toTestData.size != 0):
        # 如果测试数据不为空，输出训练数据和测试数据的形状
        print("train data size:")
        print(np.shape(toTrainData))# 输出训练数据的形状
        print("test data size:")
        print(np.shape(toTestData))# 输出测试数据的形状
        # 调用 reshape_for_save 函数将训练数据重新整形为适合计算均值和标准差的格式
        newdata = reshape_for_save(toTrainData)
        # 计算训练数据的均值，用于后续中心化
        offset = np.mean(newdata,0)
        # 计算训练数据的标准差，确保标准差最小值为 1（防止除以 0）
        scale = np.std(newdata, 0).clip(min=1)
        # 返回标准化后的训练数据和测试数据
        return rescale(toTrainData, offset, scale), rescale(toTestData, offset, scale)
    else:
        # 如果测试数据为空（例如蒸馏数据集场景）
        print("distillation data size:")
        print(np.shape(toTrainData))# 输出训练数据的形状
        # 调用 reshape_for_save 函数将训练数据重新整形为适合计算均值和标准差的格式
        newdata = reshape_for_save(toTrainData)
        # 计算训练数据的均值，用于后续中心化
        offset = np.mean(newdata,0)
        # 计算训练数据的标准差，确保标准差最小值为 1（防止除以 0）
        scale = np.std(newdata, 0).clip(min=1)
        # 返回标准化后的训练数据
        return rescale(toTrainData, offset, scale)
