import matplotlib.pyplot as plt
import json
import torch


def plot_loss_accuracy(loss, acc, round, acc_poison):

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(10,4))
    # 绘制训练损失和验证损失
    # 打印输出损失值
    plt.subplot(1, 2, 1)
    plt.plot(loss, color='blue')
    # plt.xticks(range(len(loss)))
    for x in round:
        plt.axvline(x, ymin=0, ymax=100, color='red', linestyle='dashed')
    plt.xlabel('迭代次数')
    plt.ylabel('交叉熵损失')

    plt.subplot(1, 2, 2)
    plt.plot(acc, color='orange', label='主任务正确率', linestyle='dashdot')
    plt.plot(acc_poison, color='green', label='后门正确率')
    # plt.xticks(range(len(acc)))
    for x in round:
        plt.axvline(x, ymin=0, ymax=100, color='red', linestyle='dashed')
    plt.xlabel('迭代次数')
    plt.ylabel('正确率')
    plt.legend(loc='lower right')

    # 显示图形
    plt.show()


def plot_image(data):

    with open('./utils/conf.json', 'r') as f:
        conf = json.load(f)

    # 创建一个5x5的子图网格来显示图像
    fig, axs = plt.subplots(6, 6, figsize=(10, 10))
    
    for i, ax in enumerate(axs.flat):
        # 显示图像
        image, label = data[i]
        ax.imshow(image.squeeze(), cmap='gray')
        # 设置标题为标签值
        if conf['type'] == 'fmnist':
            rel = {
                0: 'T - shirt / top',
                1: 'Trouser',
                2: 'Pullover',
                3: 'Dress',
                4: 'Coat',
                5: 'Sandal',
                6: 'Shirt',
                7: 'Sneaker',
                8: 'Bag',
                9: 'Ankle boot'
            }
            ax.set_title(str(rel[label]), loc='center', pad=1)
        elif conf['type'] == 'mnist':
            ax.set_title(str(label), loc='center', pad=1)

    # 隐藏坐标轴标签和刻度
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # 添加标题
    fig.suptitle('Poison MNIST images')

    # 显示图像
    plt.show()


def plot_accuracy(name, acc, round, acc_dis, acc_center):

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(6, 4))

    plt.plot(acc, color='orange', label='主任务正确率', linestyle='solid')
    plt.plot(acc_dis, color='green', label='后门正确率-分布式攻击', linestyle='-.')
    plt.plot(acc_center, color='blue', label='后门正确率-模型攻击', linestyle='dotted')

    # plt.xticks(range(len(acc)))
    for x in round:
        plt.axvline(x, ymin=0, ymax=100, color='red', linestyle='dashed')
    plt.xlabel('迭代次数')
    plt.ylabel('正确率')
    plt.legend(loc='lower right')
    plt.title(name)

    # 显示图形
    plt.show()

def plot_attack(name, round, acc1, acc2, acc3):

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(7, 4))

    plt.plot(acc1, color='green', label='本文算法', linestyle='solid', linewidth=2.5)
    plt.plot(acc2, color='orange', label='范数裁剪与弱差分隐私', linestyle='-.',linewidth=2.5)
    plt.plot(acc3, color='blue', label='Krum', linestyle='dotted',linewidth=2.5)

    # plt.xticks(range(len(acc)))
    for x in round:
        plt.axvline(x, ymin=0, ymax=100, color='red', linestyle='dotted',linewidth=1)
    plt.xlabel('迭代次数')
    plt.ylabel('正确率')
    plt.legend(loc='upper center')
    plt.title(name)

    # 显示图形
    plt.show()

if __name__ == '__main__':

    # data_mnist_1 = torch.load('mnist_1_defense2.pth')
    # data_mnist_4 = torch.load('mnist_4_defense_2.pth')
    # data_fmnist_1 = torch.load('fmnist_1_defense2.pth')
    # data_fmnist_4 = torch.load('fmnist_4_defense2.pth')
    # data_cifar_1 = torch.load('cifar_1_defense2.pth')
    # data_cifar_4 = torch.load('cifar_4_defense2.pth')

    data_mnist_4 = torch.load('cifar_1_defense2.pth')
    data_mnist_4_2 = torch.load('cifar_4_defense2.pth')
    # data_mnist_4_3 = torch.load('mnist_4_defense3.pth')

    # plot_attack('', [4, 19, 34, 49], data_mnist_4['acc_poison'], data_mnist_4_2['acc_poison'],data_mnist_4_3['acc_poison'])

    # plot_accuracy('C I F A R - 1 0', data_cifar_4['acc'], [4, 19, 34, 49], data_cifar_4['acc_poison'], data_cifar_1['acc_poison'])
    # plot_accuracy('M N I S T', data_mnist_4['acc'], [4, 19, 34, 49], data_mnist_4['acc_poison'], data_mnist_1['acc_poison'])
    # plot_accuracy('F M N I S T', data_fmnist_4['acc'], [4, 19, 34, 49], data_fmnist_4['acc_poison'], data_fmnist_1['acc_poison'])

    print(data_mnist_4['acc'][49], data_mnist_4['acc_poison'][49])
    print(data_mnist_4_2['acc'][49], data_mnist_4_2['acc_poison'][49])
