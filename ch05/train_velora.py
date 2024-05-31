# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_valora import TwoLayerVeLoRA

import matplotlib.pyplot as plt

import random
seed = 42
random.seed(seed)
np.random.seed(seed)

M_list = [2, 7, 14, 28]
color_list = ['blue', 'red', 'green', 'orange']
use_ema = False

for M, color in zip(M_list, color_list):
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerVeLoRA(input_size=784, hidden_size=56, output_size=10, M=M, use_ema=use_ema)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 勾配
        #grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)
        
        # 更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            iter_list.append(i)
            print(i, train_acc, test_acc)

    # last evaluation
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    iter_list.append(i)
    print(i, train_acc, test_acc)

    plt.plot(iter_list, test_acc_list, marker='.', color=color, label=f'M={M}')
plt.xlim(00, 10000)
plt.ylim(0.0, 1.0)
plt.grid()
plt.ylabel('Acc')
plt.xlabel('Step')
plt.title(f'Acc with Different Constants M\n(use_ema: {use_ema})')
plt.legend()
plt.show()