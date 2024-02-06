from network import Network
from utils import LOG_INFO
from layers import Selu, Swish, Linear, Gelu
from loss import MSELoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d

train_data, test_data, train_label, test_label = load_mnist_2d('data')

all_train_loss = {}
all_train_acc = {}
all_test_loss = {}
all_test_acc = {}
all_training_time = {}
import json
from time import time
def train_model(activate_function,loss_funtion,layernum):
    if(layernum == 1):
        model = Network()
        model.add(Linear('fc1', 784, 128, 0.01))
        model.add(activate_function('activate'))
        model.add(Linear('fc1', 128, 10, 0.01))
    if(layernum == 2):
        model = Network()
        model.add(Linear('fc1', 784, 128, 0.01))
        model.add(activate_function('activate'))
        model.add(Linear('fc1', 128, 32, 0.01))
        model.add(activate_function('activate'))
        model.add(Linear('fc1', 32, 10, 0.01))
    loss = loss_funtion(name='loss')

    # Training configuration
    # You should adjust these hyperparameters
    # NOTE: one iteration means model forward-backwards one batch of samples.
    #       one epoch means model has gone through all the training samples.
    #       'disp_freq' denotes number of iterations in one epoch to display information.

    config = {
        'learning_rate': 0.00001,
        'weight_decay': 0.1,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 200,
        'disp_freq': 50,
        'test_epoch': 1
    }

    #ADDED
    loss_list = []
    acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        #ADDED
        loss_result,acc_result = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        loss_list.append(loss_result)
        acc_list.append(acc_result)
        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            loss_result,acc_result = test_net(model, loss, test_data, test_label, config['batch_size'])
            test_loss_list.append(loss_result)
            test_acc_list.append(acc_result)

    #ADDED
    title = str(activate_function)+' '+str(loss_funtion)+' '+str(layernum)
    all_train_loss[title] = loss_list
    all_train_acc[title] = acc_list
    all_test_loss[title] = test_loss_list
    all_test_acc[title] = test_acc_list

activate_function_list = [Selu,Swish,Gelu]
loss_function_list = [MSELoss,SoftmaxCrossEntropyLoss,HingeLoss,FocalLoss]
for activate_funtion in activate_function_list:
    for loss_funtion in loss_function_list:
        for layernum in range(1,3):
            time1 = time()
            train_model(activate_funtion,loss_funtion,layernum)
            time2 = time()
            title = str(activate_funtion)+' '+str(loss_funtion)+' '+str(layernum)
            all_training_time[title] = time2-time1
            
with open("all_train_loss.json", 'w') as json_file:
    json.dump(all_train_loss, json_file,indent=4)
with open("all_train_acc.json", 'w') as json_file:
    json.dump(all_train_acc, json_file,indent=4)
with open("all_test_loss.json", 'w') as json_file:
    json.dump(all_test_loss, json_file,indent=4)
with open("all_test_acc.json", 'w') as json_file:
    json.dump(all_test_acc, json_file,indent=4)
with open("all_training_time.json", 'w') as json_file:
    json.dump(all_training_time, json_file,indent=4)
