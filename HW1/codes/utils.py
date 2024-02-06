from __future__ import division
from __future__ import print_function
import numpy as np
from datetime import datetime
#ADDED
import matplotlib.pyplot as plt

def onehot_encoding(label, max_num_class):
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding


def calculate_acc(output, label):
    correct = np.sum(np.argmax(output, axis=1) == label)
    return correct / len(label)


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)

#ADDED
def plot_info(loss_list,acc_list,test_loss_list,test_acc_list,pic_name):
    plt.figure()
    plt.plot(loss_list)
    plt.plot(test_loss_list)
    plt.legend(['train_loss','test_loss'])
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(pic_name+"_loss.png")
    plt.figure()
    plt.plot(acc_list)
    plt.plot(test_acc_list)
    plt.legend(['train_acc','test_acc'])
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(pic_name+"_acc.png")
    
