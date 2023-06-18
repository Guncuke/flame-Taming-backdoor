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
    fig = plt.figure(figsize=(5, 2))

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

def plot_noniid(tnr, tpr):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 横轴数据
    fig = plt.figure(figsize=(5, 4))

    x = [4, 19, 34, 49]

    # 纵轴数据
    y = [0, 0.5, 1]

    # 绘制两条曲线
    plt.plot(x, tpr, label='TPR', marker='s', markersize=15, color='g', linewidth=5)
    plt.plot(x, tnr, label='TNR', marker='^', markersize=17, color='k', linewidth=5)


    plt.legend(prop={'size': 20})

    # 设置横轴和纵轴的范围
    plt.xlim(min(x) - 5, max(x) + 5)
    plt.ylim(min(y) - 0.1, max(y) + 0.1)

    # 设置横轴和纵轴的刻度标签
    plt.xticks(x, ['4', '19', '34', '49'])
    plt.yticks(y, ['0%', '50%', '100%'])
    plt.xlabel('迭代次数')
    # 显示图形
    plt.show()


def plot_tpr_tnr_ba(tnr, tpr, ba):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 横轴数据
    fig = plt.figure(figsize=(9, 5))

    x = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    # 纵轴数据
    y = [0, 0.5, 1]

    # 绘制两条曲线
    plt.plot(x, tpr, label='TPR', color='g', linewidth=4, linestyle='-.', marker='o', markersize=12)
    plt.plot(x, tnr, label='TNR', color='k', linewidth=3, marker='*', markersize=15)
    plt.plot(x, ba, label='BA', color='r', linewidth=3, linestyle='--', marker='D', markersize=10)


    plt.legend(prop={'size': 15})

    # 设置横轴和纵轴的范围
    plt.xlim(min(x) - 0.5, max(x) + 0.5)
    plt.ylim(min(y) - 0.1, max(y) + 0.1)

    # 设置横轴和纵轴的刻度标签
    plt.xticks(x, ['0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'])
    plt.yticks(y, ['0%', '50%', '100%'])
    plt.xlabel(r'$\alpha$', fontsize=20)
    # 显示图形
    plt.show()

def plot_l2(l1, l2, l3):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 横轴数据
    fig = plt.figure(figsize=(10, 5))

    x = list(range(20))

    # 纵轴数据
    y = [0, 1, 2, 3, 4]

    # 绘制两条曲线
    plt.plot(x, l1, label='MNIST', color='g', linewidth=4, linestyle='-.', marker='o', markersize=10)
    plt.plot(x, l2, label='FMNIST', color='k', linewidth=3, marker='x', markersize=12)
    plt.plot(x, l3, label='CIFAR10', color='orange', linewidth=3, linestyle='--', marker='D', markersize=10)


    plt.legend(prop={'size': 20})

    # 设置横轴和纵轴的范围
    plt.xlim(min(x) - 0.5, max(x) + 0.5)
    plt.ylim(min(y) - 0.1, max(y) + 0.1)

    # 设置横轴和纵轴的刻度标签
    plt.xticks(x, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
    plt.yticks(y, ['0', '1', '2', '3', '4'])
    plt.xlabel('迭代轮次', fontsize=15)
    plt.ylabel(r'$L_2$范数', fontsize=15)
    # 显示图形
    plt.show()
