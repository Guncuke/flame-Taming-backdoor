import matplotlib.pyplot as plt
import json


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
    plt.title('交叉熵损失')
    plt.xlabel('迭代次数')
    plt.ylabel('交叉熵损失')

    plt.subplot(1, 2, 2)
    plt.plot(acc, color='orange', label='主任务正确率', linestyle='dashdot')
    plt.plot(acc_poison, color='green', label='后门正确率')
    # plt.xticks(range(len(acc)))
    for x in round:
        plt.axvline(x, ymin=0, ymax=100, color='red', linestyle='dashed')
    plt.title('正确率')
    plt.xlabel('迭代次数')
    plt.ylabel('正确率')
    plt.legend(loc='lower left')

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
                0 : 'T - shirt / top',
                1 : 'Trouser',
                2 : 'Pullover',
                3 : 'Dress',
                4 : 'Coat',
                5 : 'Sandal',
                6 : 'Shirt',
                7 : 'Sneaker',
                8 : 'Bag',
                9 : 'Ankle boot'
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
