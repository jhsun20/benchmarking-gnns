"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import json
from typing import Union
from ray.train import report
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.MC_node_classification.load_net import gnn_model  # import GNNs
from data.data import LoadData  # import dataset

_root = os.getcwd()
"""
    GPU Setup
"""


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available() and use_gpu:
        print(f'CUDA available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    #print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        #print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_test_pipeline(model_name, train_dataset, val_dataset, params, net_params, setup, dirs, test=False, test_dataset=None):

    start0 = time.time()
    per_epoch_time = []

    dataset_name = train_dataset.name

    if model_name in ['GCN', 'GAT']:
        if net_params['self_loop']:
            train_dataset._add_self_loops()
            # test_dataset._add_self_loops()

    root_log_dir, root_ckpt_dir, write_file_name, write_file_name_test, losses_dir = dirs

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # set device
    device = net_params['device']

    # setting seeds
    if params['seed'] != "random":
        seed = params['seed']
    else:
        seed = random.randint(1, 100)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    print("Training Graphs: ", len(train_dataset.dataset))
    print("Validation Graphs: ", len(val_dataset.dataset))
    print("Number of Classes: ", net_params['n_classes'])

    model = gnn_model(model_name, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []
    epoch_train_f1s, epoch_val_f1s = [], []

    # import train functions for all other GCNs
    from train.train_MVC_node_classification import train_epoch, train_epoch_all_optimal, evaluate_network, evaluate_network_all_optimal

    # train_loader = DataLoader(train_dataset.dataset, batch_size=net_params['batch_size'], shuffle=True, collate_fn=train_dataset.collate)
    # val_loader = DataLoader(val_dataset.dataset, batch_size=net_params['batch_size'], shuffle=False, collate_fn=train_dataset.collate)
    train_loader = DataLoader(train_dataset.dataset, batch_size=1, shuffle=True, collate_fn=train_dataset.collate)
    val_loader = DataLoader(val_dataset.dataset, batch_size=1, shuffle=False, collate_fn=train_dataset.collate)

    with tqdm(range(params['epochs'])) as t:
        for epoch in t:

            t.set_description('Epoch %d' % epoch)

            start = time.time()
            # epoch_train_loss, epoch_train_acc, epoch_train_f1, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
            epoch_train_loss, epoch_train_acc, epoch_train_f1, optimizer = train_epoch_all_optimal(model, optimizer,
                                                                                                   device, train_loader,
                                                                                                   epoch, net_params[
                                                                                                       'batch_size'])

            epoch_val_loss, epoch_val_acc, epoch_val_f1 = evaluate_network_all_optimal(model, device, val_loader)

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            epoch_train_accs.append(epoch_train_acc)
            epoch_val_accs.append(epoch_val_acc)
            epoch_train_f1s.append(epoch_train_f1)
            epoch_val_f1s.append(epoch_val_f1)

            writer.add_scalar('train/_loss', epoch_train_loss, epoch)
            writer.add_scalar('val/_loss', epoch_val_loss, epoch)
            writer.add_scalar('train/_acc', epoch_train_acc, epoch)
            writer.add_scalar('val/_acc', epoch_val_acc, epoch)
            writer.add_scalar('train/_f1', epoch_train_f1, epoch)
            writer.add_scalar('val/_f1', epoch_val_f1, epoch)
            # writer.add_scalar('test/_acc', epoch_test_acc, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            report({
                "epoch_train_loss":epoch_train_loss, 
                "epoch_val_loss":epoch_val_loss, 
                "epoch_train_acc":epoch_train_acc, 
                "epoch_val_acc":epoch_val_acc,
                "epoch_train_f1":epoch_train_f1, 
                "epoch_val_f1":epoch_val_f1
                })

            t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                          train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                          train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                          train_f1=epoch_train_f1, val_f1=epoch_val_f1)

            per_epoch_time.append(time.time() - start)

            # Saving checkpoint
            ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

            files = glob.glob(ckpt_dir + '/*.pkl')
            for file in files:
                epoch_nb = file.split('_')[-1]
                epoch_nb = int(epoch_nb.split('.')[0])
                if epoch_nb < epoch - 1:
                    os.remove(file)

            scheduler.step(epoch_val_loss)

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                break

            # Stop training after params['max_time'] hours
            if time.time() - start0 > params['max_time'] * 3600:
                print('-' * 89)
                print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                break

    _, val_acc, val_f1 = evaluate_network_all_optimal(model, device, val_loader)
    _, train_acc, train_f1 = evaluate_network_all_optimal(model, device, train_loader)
    print("Val Accuracy: {:.4f}".format(val_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Val F1: {:.4f}".format(val_f1))
    print("Train F1: {:.4f}".format(train_f1))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - start0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    report({
        "val_acc":val_acc, 
        "train_acc":train_acc, 
        "val_f1":val_f1, 
        "train_f1":train_f1, 
        "total_epoch":epoch
        })
    writer.close()

    """
        Write the results in out_dir/results folder and save losses/accs
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nVAL ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\nVAL F1: {:.4f}\nTRAIN F1: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                .format(dataset_name, model_name, params, net_params, model, net_params['total_param'],
                        val_acc, train_acc, val_f1, train_f1, epoch, (time.time() - start0) / 3600, np.mean(per_epoch_time)))

    os.makedirs(losses_dir, exist_ok=True)
    with open(losses_dir + 'train.json', 'w+') as output_file:
        json.dump([epoch_train_accs, epoch_train_losses], output_file, indent=4)
    with open(losses_dir + 'val.json', 'w+') as output_file:
        json.dump([epoch_val_accs, epoch_val_losses], output_file, indent=4)

    if test:
        dataset_name = test_dataset.name
        test_loader = DataLoader(test_dataset.dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate)

        _, test_acc, test_f1 = evaluate_network_all_optimal(model, device, test_loader)
        print("Test Accuracy: {:.4f}".format(test_acc))
        print("Test F1: {:.4f}".format(test_f1))
        print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - start0))

        """
            Write the results in out_dir/results folder
        """
        with open(write_file_name_test + '.txt', 'w') as f:
            f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\nTrained on {}\n\n
                     FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTEST F1: {:.4f}\nTotal Time Taken: {:.4f} hrs\n\n\n"""
                    .format(dataset_name, model_name, params, net_params, model, net_params['total_param'],
                            setup['train_dataset'],
                            test_acc, test_f1, (time.time() - start0) / 3600))


def test_pipeline(model_name, test_dataset, weights_path, params, net_params, setup, dirs):
    start0 = time.time()

    dataset_name = test_dataset.name

    if model_name in ['GCN', 'GAT']:
        if net_params['self_loop']:
            test_dataset._add_self_loops()

    write_file_name = dirs
    device = net_params['device']

    print("Test Graphs: ", len(test_dataset.dataset))
    print("Number of Classes: ", net_params['n_classes'])

    model = gnn_model(model_name, net_params)
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)

    # import train functions for all other GCNs
    from train.train_MC_node_classification import evaluate_network_all_optimal

    test_loader = DataLoader(test_dataset.dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate)

    _, test_acc, test_f1 = evaluate_network_all_optimal(model, device, test_loader)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Test F1: {:.4f}".format(test_f1))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - start0))

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\nTrained on {}\n\n
                 FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTEST F1: {:.4f}\nTotal Time Taken: {:.4f} hrs\n\n\n"""
                .format(dataset_name, model_name, params, net_params, model, net_params['total_param'],
                        setup['train_dataset'],
                        test_acc, test_f1, (time.time() - start0) / 3600))


def train(config_path:Union[str,dict]):
    """
    train model
    """
    if isinstance(config_path,str):
        with open(config_path) as f:
            config = json.load(f)
    elif isinstance(config_path,dict):
        config = config_path
    # set up device
    device = gpu_setup(config['setup']['gpu']['use'], config['setup']['gpu']['id'])
    config['setup']['gpu']['id'] = device.index
    # setup model, train dataset, output directory
    model = config['setup']['model']
    dataset_name = config['setup']['train_dataset']
    out_dir = os.path.join(_root,config['setup']['out_dir'])
    features = config['setup']['features']
    # save config file
    write_config_file = out_dir + 'configs/config_' + model + "_" + dataset_name + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_log_dir = out_dir + 'logs/' + model + "_" + dataset_name + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + model + "_" + dataset_name + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name_train = out_dir + 'results/TRAIN_result_' + model + "_" + dataset_name + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name_test = out_dir + 'results/TEST_result_' + model + "_" + dataset_name + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    losses_dir = out_dir + 'losses/' + model + "_" + dataset_name + "_GPU" + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y') + '/'
    dirs = root_log_dir, root_ckpt_dir, write_file_name_train, write_file_name_test, losses_dir
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    os.makedirs(out_dir + 'configs', exist_ok=True)
    with open(write_config_file + '.json', 'w') as f:
        json.dump(config, f, indent=4)

    # load dataset
    train_dataset = LoadData(data_dir=os.path.join(_root, 'data/CO/train'), name=dataset_name, split='train', features=features)
    val_dataset = LoadData(data_dir=os.path.join(_root,'data/CO/val'), name=dataset_name, split='val', features=features)
    if config['setup']['test_dataset'] != "none":
        test_dataset = LoadData(data_dir=os.path.join(_root,'data/CO/test'), name=config['setup']['test_dataset'], split='test', features=features)
    else:
        test_dataset = None
    sample = train_dataset.dataset[0][0].ndata['feat']
    # print("sample graph node features")
    # print(sample)
    # set up parameters
    setup = config['setup']
    params = {**config['params'], **config['tunable_params']}
    node_in_dim = sample.size(dim=1)
    net_params = {**config['net_params'], **config['tunable_net_params'], 'in_dim': node_in_dim, 'device': device}
    net_params['n_classes'] = 2
    net_params['edge_dim'] = 1
    net_params['total_param'] = view_model_param(model, net_params)

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    train_test_pipeline(model, train_dataset, val_dataset, params, net_params, setup, dirs, test=False, test_dataset=test_dataset)


def test(config_path: Union[str,dict]):
    """
    test model
    """
    if isinstance(config_path,str):
        with open(config_path) as f:
            config = json.load(f)
    elif isinstance(config_path,dict):
        config = config_path
    # get OLD parameters for the saved model
    weights_path = os.path.join(_root, config['setup']['saved_weights'])
    with open(config['setup']['saved_config']) as f:
        train_config = json.load(f)
    # set up device
    device = gpu_setup(config['setup']['gpu']['use'], config['setup']['gpu']['id'])
    # setup model, train dataset, output directory
    config['setup']['gpu']['id'] = device.index
    model = config['setup']['model']
    dataset_name = config['setup']['test_dataset']
    out_dir = os.path.join(_root,config['setup']['out_dir'])
    features = train_config['setup']['features']
    # load test set
    test_dataset = LoadData(data_dir=os.path.join(_root, 'data/CO/test'), name=dataset_name, split='test', features=features)
    sample = test_dataset.dataset[0][0].ndata['feat']
    setup = train_config['setup']
    params = {**train_config['params'], **train_config['tunable_params']}
    node_in_dim = sample.size(dim=1)
    net_params = {**train_config['net_params'], **train_config['tunable_net_params'], 'in_dim': node_in_dim, 'device': device}
    print('in dim:', net_params['in_dim'])
    net_params['n_classes'] = 2
    net_params['edge_dim'] = 1
    net_params['total_param'] = view_model_param(model, net_params)

    write_file_name_test = out_dir + 'results/TEST_result_' + model + "_" + dataset_name + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs_test = write_file_name_test
    test_pipeline(model, test_dataset, weights_path, params, net_params, setup, dirs_test)

__train_config_dict__ = {
    'GAT': 'configs/MC/MC_GAT_100k_train_base.json',
    'EGT': 'configs/MC/MC_EGT_100k_train_base.json',
    'GCN': 'configs/MC/MC_GCN_100k_train_base.json',
    'GIN': 'configs/MC/MC_GIN_100k_train_base.json',
    'GMM': 'configs/MC/MC_GMM_100k_train_base.json',
    'GraphSage': 'configs/MC/MC_GraphSage_100k_train_base.json',
    'GatedGCN': 'configs/MC/MC_GatedGCN_100k_train_base.json',
    'PNA': 'configs/MC/MC_PNA_100k_train_base.json'
}

__test_config_dict__ = {
    'GAT': 'configs/MC/test/MC_GAT_100k_test_base.json',
    'EGT': 'configs/MC/test/MC_EGT_100k_test_base.json',
    'GCN': 'configs/MC/test/MC_GCN_100k_test_base.json',
    'GIN': 'configs/MC/test/MC_GIN_100k_test_base.json',
    'GMM': 'configs/MC/test/MC_GMM_100k_test_base.json',
    'GraphSage': 'configs/MC/test/MC_GraphSage_100k_test_base.json',
    'GatedGCN': 'configs/MC/test/MC_GatedGCN_100k_test_base.json',
    'PNA': 'configs/MC/test/MC_PNA_100k_test_base.json'
}
def main():

    config_path = __train_config_dict__['PNA']
    train(config_path=config_path)
    
    # LOAD NEW CONFIG IF NEED TO
    config_path = __test_config_dict__['GAT']
    #test(config_path=config_path)


if __name__ == '__main__':
    main()
