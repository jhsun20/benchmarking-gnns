# %%
import ray
from ray import tune
from ray.tune import TuneConfig
from ray.train import RunConfig, FailureConfig
from ray.tune.search.hyperopt import HyperOptSearch
from main_MC_node_classification import train as train_MC
from main_MIS_node_classification import train as train_MIS
from main_MVC_node_classification import train as train_MVC
import argparse
import os
import json
from pathlib import Path

# define hyperparameter space
def get_search_space_only_loss(model: str,dataset:str):
    search_space = {}
    
    search_space['setup'] = {
        "gpu": {
            "use": True,
            "id": -1
        },
        "model": model,
        "train_dataset": dataset,
        "val_dataset": dataset,
        "test_dataset": "none",
        "out_dir": f"out/{dataset}_node_classification/",
        "features": "basic"
    }

    search_space["params"] = {
        "seed": "random",
        "epochs": 100,
        "print_epoch_interval": 1,
        "max_time": 24
    }

    search_space["tunable_params"] = {
        "init_lr": 1e-2,
        "lr_reduce_factor": 0.5,
        "lr_schedule_patience": 10,
        "min_lr": 0.0005,
        "weight_decay": 0.0005
    }

    if model == 'EGT': 
        search_space["net_params"] = {
            "L": 4,
            "node_hidden_dim": 60,
            "n_heads": 6,
            "edge_hidden_dim": 1,
            "edge_update": False,
            "graph_size": 30,
            "num_virtual_nodes": 0
        }
        search_space["tunable_net_params"] = {
            "batch_size": 32,
            "dropout": 0.2,
            "attn_drop": 0.2,
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
    if model == 'GAT':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 12,
            "out_dim": 120,
            "n_heads": 8
        }
        search_space["tunable_net_params"] = {
            "batch_size": 32,
            "batch_norm": True,
            "dropout": 0.2,
            "attn_drop": 0.2,
            "residual": True,
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
    if model == 'GatedGCN':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 65,
            "out_dim": 65
        }
        search_space["tunable_net_params"] = {
            "batch_size": 32,
            "batch_norm": True,
            "dropout": 0.2,
            "residual":True,
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
    if model == 'GCN':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 140,
            "out_dim": 140
        }
        search_space["tunable_net_params"] = {
            "batch_size": 32,
            "batch_norm": True,
            "dropout": 0.2,
            "residual": True,
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
    if model == 'GIN':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 105,
            "out_dim": 105,
            "apply_fn_layers": 2,
            "learn_eps": True,
            "aggr_type": "max"
        }
        search_space["tunable_net_params"] = {
            "batch_size": 64,
            "batch_norm": True,
            "dropout": 0.2,
            "residual":True,
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
    if model == 'GMM':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 105,
            "out_dim": 105,
            "aggr_type": "max",
            "pseudo_dim": 3,
            "kernel": 2
        }
        search_space["tunable_net_params"] =  {
            "batch_size": 32,
            "batch_norm": True,
            "dropout": 0.2,
            "residual": True,
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
    if model == 'GraphSage':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 45,
            "out_dim": 45,
            "sage_aggregator": "lstm"
        }
        search_space["tunable_net_params"] = {
            "batch_size": 64,
            "batch_norm": True,
            "dropout": 0.2,
            "residual": True,
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
        
    return search_space


def get_search_space(model: str,dataset:str):
    search_space = {}
    
    search_space['setup'] = {
        "gpu": {
            "use": True,
            "id": -1
        },
        "model": model,
        "train_dataset": dataset,
        "val_dataset": dataset,
        "test_dataset": "none",
        "out_dir": f"out/{dataset}_node_classification/",
        "features": "basic"
    }

    search_space["params"] = {
        "seed": "random",
        "epochs": 100,
        "print_epoch_interval": 1,
        "max_time": 24
    }

    search_space["tunable_params"] = {
        "init_lr": tune.loguniform(1e-3, 1e-1),
        "lr_reduce_factor": tune.choice([0.2,0.5,0.8]),
        "lr_schedule_patience": 10,
        "min_lr": 1e-4,
        "weight_decay": tune.loguniform(1e-5, 1e-3)
    }

    if model == 'EGT': 
        search_space["net_params"] = {
            "L": 4,
            "node_hidden_dim": 60,
            "n_heads": 6,
            "edge_hidden_dim": 1,
            "edge_update": False,
            "graph_size": 30,
            "num_virtual_nodes": 0
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "attn_drop": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)],
        }
    if model == 'GAT':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 15,
            "out_dim": 120,
            "n_heads": 8
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "attn_drop": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "neg_slope": tune.choice([0.0,0.2,0.4,0.8]),
            "residual": tune.choice([True, False]),
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
    if model == 'GatedGCN':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 65,
            "out_dim": 65
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "residual": tune.choice([True, False]),
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
    if model == 'GCN':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 140,
            "out_dim": 140
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "residual": tune.choice([True, False]),
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
    if model == 'GIN':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 105,
            "out_dim": 105,
            "apply_fn_layers": 2,
            "learn_eps": True,
            "aggr_type": "max"
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "residual": tune.choice([True, False]),
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)],
        }
    if model == 'GMM':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 105,
            "out_dim": 105,
            "aggr_type": "max",
            "pseudo_dim": 3,
            "kernel": 2
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "residual": tune.choice([True, False]),
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)],
        }
    if model == 'GraphSage':
        search_space["net_params"] = {
            "L": 4,
            "hidden_dim": 45,
            "out_dim": 45,
            "sage_aggregator": "lstm"
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "residual": tune.choice([True, False]),
            "loss_weight": [1.0, tune.loguniform(1e-1,1e1)]
        }
        
    return search_space


_datasets = ['MC', 'MVC', 'MIS']
_models = ['EGT','GAT','GatedGCN','GCN','GIN','GMM','GraphSage']



def main(args):
    reporter = tune.CLIReporter(
        metric_columns=["val_f1"],
        parameter_columns=["setup/model","setup/train_dataset", "tunable_net_params/loss_weight"],
        print_intermediate_tables=False,
        max_column_length=12,
        max_report_frequency=30,
        max_progress_rows=100
    )
    tune_configer= TuneConfig(
        max_concurrent_trials=12,  # todo
        num_samples=10,
        search_alg=HyperOptSearch(metric="val_f1", mode="max"),
        trial_name_creator=lambda trial: f"{trial.trial_id}_{trial.config['setup']['model']}_{trial.config['setup']['train_dataset']}_{trial.config['tunable_net_params']['loss_weight'][1]:.3f}"
    )
    log_dir = f"{args.dataset}_{args.model}"
    os.makedirs(f"/home/xiongzhixiao/ray_results/{log_dir}", exist_ok=True)
    
    run_configer = RunConfig(
        name=log_dir,
        storage_path=f"/home/xiongzhixiao/ray_results/{log_dir}",
        progress_reporter=reporter,
        stop={"time_total_s":7200},
        failure_config=FailureConfig(fail_fast=False),
    )
    train_func = {
        "MC": train_MC,
        "MVC": train_MVC,
        "MIS": train_MIS
    }[args.dataset]
    
    if args.only_loss:
        search_space = get_search_space_only_loss(args.model,args.dataset)
    else:
        search_space = get_search_space(args.model,args.dataset)
        
    tuner = tune.Tuner(
        trainable=tune.with_resources(trainable=train_func, resources={"cpu": args.gpu, "gpu": args.gpu} if args.gpu > 0 else {"cpu": args.cpu}),
        param_space=search_space,
        run_config=run_configer,
        tune_config=tune_configer
    )

    results = tuner.fit()
    best_config = results.get_best_result(metric="val_f1", mode="max")
    with open(f"out/{args.dataset}_node_classification/{args.dataset}_{args.model}_best.json", "w") as f:
        json.dump(best_config.config, f, indent=4)
    df = results.get_dataframe()
    df.to_csv(f"out/{args.dataset}_node_classification/{args.dataset}_{args.model}.csv")
    print(f"Best config: {best_config}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',"-m", type=str, required=True, help='model name', choices=_models)
    parser.add_argument('--dataset', "-d", type=str, required=True, help='dataset name', choices=_datasets)
    parser.add_argument('--device', type=str, default="0", help='gpu ids (str)')
    parser.add_argument('--cpu', "-c", type=float, default=1.0, help='cpu per trial')
    parser.add_argument('--gpu', "-g", type=float, default=0.5, help='gpu per trial')
    parser.add_argument('--only_loss', '-l', action='store_true', help='only tune loss weight')
    return parser.parse_args()
# %%
if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["RAY_memory_usage_threshold"] = "0.98"
    main(args)

# %%
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt 
#     import itertools
#     import numpy as np
#     results_dir = Path("/home/xiongzhixiao/ray_results")  # todo
#     datasets = ["MIS", "MVC", "MC"]
#     models = ["EGT", "GAT", "GatedGCN", "GCN", "GIN", "GMM", "GraphSage"]
#     dir_names = [f"{dataset}_{model}" for dataset, model in itertools.product(datasets, models)]
#     for d in results_dir.iterdir():
#         if d.is_dir() and d.name in dir_names:
#             dataset, model = d.name.split("_")
#             train_func = {
#                 "MC": train_MC,
#                 "MVC": train_MVC,
#                 "MIS": train_MIS
#             }[dataset]
#             restored_tuner = tune.Tuner.restore(str(d),trainable=train_func)
#             result_grid = restored_tuner.get_results()
#             best_config = result_grid.get_best_result(metric="val_f1", mode="max")
#             loss_weights = np.array([result.config['tunable_net_params']['loss_weight'][1] for result in result_grid])
#             val_f1s = np.array([result.metrics['val_f1'] for result in result_grid])
#             # sort loss_weights and val_f1s according to loss_weights in ascending order
#             idx = np.argsort(loss_weights)
#             loss_weights = loss_weights[idx]
#             val_f1s = val_f1s[idx]
#             plt.figure(figsize=(8, 6))
#             plt.scatter(loss_weights, val_f1s, marker='o', s=100, alpha=0.5)
#             plt.plot(loss_weights, val_f1s, label='raw curve', color='red',alpha=0.5)
#             plt.xlabel("loss weight (1:x)")
#             plt.ylabel("valation f1 score")
#             plt.title(f"Tuning results of {dataset} on {model}")
#             plt.legend()
#             # add other configs to the plot
#             plt.text(0.5, 0.95, f"best loss weight: {best_config.config['tunable_net_params']['loss_weight'][1]:.3f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='center')
#             plt.tight_layout()
#             plt.savefig(f"out/{dataset}_node_classification/{dataset}_{model}.png")