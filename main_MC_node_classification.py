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

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.CO_node_classification.load_net import gnn_model  # import GNNs
from data.data import LoadData  # import dataset

"""
    GPU Setup
"""


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
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
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_val_pipeline(MODEL_NAME, train_dataset, val_dataset, params, net_params, dirs):
    start0 = time.time()
    per_epoch_time = []

    DATASET_NAME = train_dataset.name

    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            train_dataset._add_self_loops()
            # test_dataset._add_self_loops()

    if MODEL_NAME in ['GatedGCN']:
        if net_params['pos_enc']:
            print("[!] Adding graph positional encoding.")
            train_dataset._add_positional_encodings(net_params['pos_enc_dim'])
            # test_dataset._add_positional_encodings(net_params['pos_enc_dim'])
            print('Time PE:', time.time() - start0)

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file, losses_dir = dirs
    device = net_params['device']

    # Write network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format
                (DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Training Graphs: ", len(train_dataset.dataset))
    print("Validation Graphs: ", len(val_dataset.dataset))
    # print("Test Graphs: ", len(testset))
    print("Number of Classes: ", net_params['n_classes'])

    model = gnn_model(MODEL_NAME, net_params)
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
    from train.train_CO_node_classification import train_epoch, evaluate_network

    train_loader = DataLoader(train_dataset.dataset, batch_size=params['batch_size'], shuffle=True,
                              collate_fn=train_dataset.collate)
    val_loader = DataLoader(val_dataset.dataset, batch_size=params['batch_size'], shuffle=False,
                            collate_fn=train_dataset.collate)
    # test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=test_dataset.collate)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()
                epoch_train_loss, epoch_train_acc, epoch_train_f1, optimizer = train_epoch(model, optimizer, device, train_loader,
                                                                           epoch)

                epoch_val_loss, epoch_val_acc, epoch_val_f1 = evaluate_network(model, device, val_loader)

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

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    _, val_acc, val_f1 = evaluate_network(model, device, val_loader)
    _, train_acc, train_f1 = evaluate_network(model, device, train_loader)
    print("Val Accuracy: {:.4f}".format(val_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Val F1: {:.4f}".format(val_f1))
    print("Train F1: {:.4f}".format(train_f1))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - start0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder and save losses/accs
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nVAL ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\nVAL F1: {:.4f}\nTRAIN F1: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        val_acc, train_acc, val_f1, train_f1, epoch, (time.time() - start0) / 3600, np.mean(per_epoch_time)))

    os.makedirs(losses_dir, exist_ok=True)
    with open(losses_dir + 'train.json', 'w+') as output_file:
        json.dump([epoch_train_accs, epoch_train_losses], output_file, indent=2)
    with open(losses_dir + 'val.json', 'w+') as output_file:
        json.dump([epoch_val_accs, epoch_val_losses], output_file, indent=2)


def test_pipeline(MODEL_NAME, test_dataset, params, net_params, dirs):
    start0 = time.time()

    DATASET_NAME = test_dataset.name

    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            test_dataset._add_self_loops()

    if MODEL_NAME in ['GatedGCN']:
        if net_params['pos_enc']:
            print("[!] Adding graph positional encoding.")
            test_dataset._add_positional_encodings(net_params['pos_enc_dim'])
            print('Time PE:', time.time() - start0)

    write_file_name = dirs
    device = net_params['device']

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Test Graphs: ", len(test_dataset.dataset))
    print("Number of Classes: ", net_params['n_classes'])

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    # import train functions for all other GCNs
    from train.train_CO_node_classification import evaluate_network_all_optimal

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
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        params['train_set'],
                        test_acc, test_f1, (time.time() - start0) / 3600))


def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--train_dataset', help="Please give a value for train dataset name")
    parser.add_argument('--val_dataset', help="Please give a value for val dataset name")
    parser.add_argument('--test_dataset', help="Please give a value for test dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--pos_enc', help="Please give a value for pos_enc")
    args = parser.parse_args()
    with open('configs/MC_node_classification_GAT_100k.json') as f:
        config = json.load(f)
    print(torch.cuda.is_available())
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.train_dataset is not None:
        TRAIN_DATASET_NAME = args.train_dataset
    else:
        TRAIN_DATASET_NAME = config['train_dataset']
    train_dataset = LoadData(data_dir='data/CO/train', name=TRAIN_DATASET_NAME, split='train', features="degree")
    val_dataset = LoadData(data_dir='data/CO/val', name=TRAIN_DATASET_NAME, split='val', features="degree")
    print(train_dataset.dataset[0][0].ndata['feat'])
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated == 'True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred == 'True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat == 'True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False
    if args.pos_enc is not None:
        net_params['pos_enc'] = True if args.pos_enc == 'True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)

    # CO
    net_params['in_dim'] = 2
    print('in dim:', net_params['in_dim'])
    net_params['n_classes'] = 2
    #net_params['edge_dim'] = 1
    params['train_set'] = TRAIN_DATASET_NAME
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name_train = out_dir + 'results/TRAIN_result_' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    losses_dir = out_dir + 'losses/' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y') + '/'
    dirs = root_log_dir, root_ckpt_dir, write_file_name_train, write_config_file, losses_dir

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)

    train_val_pipeline(MODEL_NAME, train_dataset, val_dataset, params, net_params, dirs)

    if args.test_dataset is not None:
        TEST_DATASET_NAME = args.test_dataset
    else:
        TEST_DATASET_NAME = config['test_dataset']
    test_dataset = LoadData(data_dir='data/CO/test', name=TEST_DATASET_NAME, split='test')
    write_file_name_test = out_dir + 'results/TEST_result_' + MODEL_NAME + "_" + TEST_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs_test = write_file_name_test
    test_pipeline(MODEL_NAME, test_dataset, params, net_params, dirs_test)

    # ---------------------------------    ANOTHER RUN       -------------------------------------
    params['seed'] = 12
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name_train = out_dir + 'results/TRAIN_result_' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    losses_dir = out_dir + 'losses/' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y') + '/'
    dirs = root_log_dir, root_ckpt_dir, write_file_name_train, write_config_file, losses_dir
    train_val_pipeline(MODEL_NAME, train_dataset, val_dataset, params, net_params, dirs)

    if args.test_dataset is not None:
        TEST_DATASET_NAME = args.test_dataset
    else:
        TEST_DATASET_NAME = config['test_dataset']
    test_dataset = LoadData(data_dir='data/CO/test', name=TEST_DATASET_NAME, split='test')
    write_file_name_test = out_dir + 'results/TEST_result_' + MODEL_NAME + "_" + TEST_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs_test = write_file_name_test
    test_pipeline(MODEL_NAME, test_dataset, params, net_params, dirs_test)

    # ---------------------------------    ANOTHER RUN       -------------------------------------
    params['seed'] = 45
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name_train = out_dir + 'results/TRAIN_result_' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    losses_dir = out_dir + 'losses/' + MODEL_NAME + "_" + TRAIN_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y') + '/'
    dirs = root_log_dir, root_ckpt_dir, write_file_name_train, write_config_file, losses_dir
    train_val_pipeline(MODEL_NAME, train_dataset, val_dataset, params, net_params, dirs)

    if args.test_dataset is not None:
        TEST_DATASET_NAME = args.test_dataset
    else:
        TEST_DATASET_NAME = config['test_dataset']
    test_dataset = LoadData(data_dir='data/CO/test', name=TEST_DATASET_NAME, split='test')
    write_file_name_test = out_dir + 'results/TEST_result_' + MODEL_NAME + "_" + TEST_DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs_test = write_file_name_test
    test_pipeline(MODEL_NAME, test_dataset, params, net_params, dirs_test)


main()
