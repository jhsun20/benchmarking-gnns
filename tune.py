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

# %%
# define hyperparameter space

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
        "epochs": 500,
        "print_epoch_interval": 1,
        "max_time": 24
    }

    search_space["tunable_params"] = {
        "init_lr": tune.loguniform(1e-3, 1e-1),
        "lr_reduce_factor": tune.choice([0.2,0.5,0.8]),
        "lr_schedule_patience": 10,
        "min_lr": 1e-5,
        "weight_decay": tune.loguniform(1e-5, 1e-3)
    }

    if model == 'EGT': 
        search_space["net_params"] = {
            "L": 4,
            "edge_hidden_dim": 1,
            "edge_update": False,
            "graph_size": 30,
            "num_virtual_nodes": 0
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "attn_drop": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "loss_weight": [tune.loguniform(1e-1,1e1),1.0],
            "node_hidden_dim": tune.choice([32, 64, 128]),
            "n_heads": tune.choice([4, 8, 16]),
        }
    if model == 'GAT':
        search_space["net_params"] = {
            "L": 4
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "hidden_dim": 15,
            "out_dim": 120,
            "n_heads": 8,
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "attn_drop": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "neg_slope": tune.choice([0.0,0.2,0.4,0.8]),
            "residual": tune.choice([True, False]),
            "self_loop": tune.choice([True, False]),
            "loss_weight": [tune.loguniform(1e-1,1e1),1.0]
        }
    if model == 'GatedGCN':
        search_space["net_params"] = {
            "L": 4
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "hidden_dim": tune.choice([32, 64, 128]),
            "out_dim": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "neg_slope": tune.choice([0.0,0.2,0.4,0.8]),
            "residual": tune.choice([True, False]),
            "loss_weight": [tune.loguniform(1e-1,1e1),1.0]
        }
    if model == 'GCN':
        search_space["net_params"] = {
            "L": 4
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "hidden_dim": tune.choice([32, 64, 128]),
            "out_dim": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "residual": tune.choice([True, False]),
            "self_loop": tune.choice([True, False]),
            "loss_weight": [tune.loguniform(1e-1,1e1),1.0]
        }
    if model == 'GIN':
        search_space["net_params"] = {
            "L": 4
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "hidden_dim": tune.choice([32, 64, 128]),
            "out_dim": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "residual": tune.choice([True, False]),
            "loss_weight": [tune.loguniform(1e-1,1e1),1.0],
            "apply_fn_layers": tune.choice([1, 2, 3]),
            "learn_eps": tune.choice([True, False]),
            "aggr_type": tune.choice(['sum', 'max','mean'])
        }
    if model == 'GMM':
        search_space["net_params"] = {
            "L": 4
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "hidden_dim": tune.choice([32, 64, 128]),
            "out_dim": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "neg_slope": tune.choice([0.0,0.2,0.4,0.8]),
            "residual": tune.choice([True, False]),
            "loss_weight": [tune.loguniform(1e-1,1e1),1.0],
            "aggr_type": tune.choice(['sum', 'max','mean']),
            "pseudo_dim": tune.choice([2,3,4]),
            "kernel": tune.choice([1,2,3,4])
        }
    if model == 'GraphSage':
        search_space["net_params"] = {
            "L": 4
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "hidden_dim": tune.choice([32, 64, 128]),
            "out_dim": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "sage_aggregator": tune.choice(['mean', 'gcn', 'pool', 'lstm']),
            "residual": tune.choice([True, False]),
            "loss_weight": [tune.loguniform(1e-1,1e1),1.0]
        }
    if model == 'PNA':
        search_space["net_params"] = {
            "L": 4,
            "aggregators": ["mean", "max", "min", "std", "var", "sum", "moment3"],
            "scalers": ["identity", "amplification", "attenuation"],
            "num_towers": 1
        }
        search_space["tunable_net_params"] = {
            "batch_size": tune.choice([16, 32, 64]),
            "hidden_dim": tune.choice([32, 64, 128]),
            "out_dim": tune.choice([16, 32, 64]),
            "batch_norm": tune.choice([True, False]),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "residual": tune.choice([True, False]),
            "loss_weight": [tune.loguniform(1e-1,1e1),1.0],
            "delta": tune.uniform(0.0, 10.0),
        }
        
    return search_space


_datasets = ['MC', 'MVC', 'MIS']
_models = ['EGT','GAT','GAtedGCN','GCN','GIN','GMM','GraphSage','PNA']



def main(args):
    reporter = tune.CLIReporter(
        metric_columns=["val_f1"],
        parameter_columns=["setup/model","setup/dataset", "tunable_net_params/batch_size", "tunable_net_params/loss_weight"],
        print_intermediate_tables=False,
        max_column_length=12,
        max_report_frequency=30,
        max_progress_rows=100
    )
    tune_configer= TuneConfig(
        max_concurrent_trials=12,  # todo
        num_samples=20,
        search_alg=HyperOptSearch(metric="val_f1", mode="max")
    )
    run_configer = RunConfig(
        progress_reporter=reporter,
        # verbose=1,
        stop={"time_total_s":7200},
        failure_config=FailureConfig(fail_fast=False)
    )
    train_func = {
        "MC": train_MC,
        "MVC": train_MVC,
        "MIS": train_MIS
    }[args.dataset]
    
    tuner = tune.Tuner(
        trainable=tune.with_resources(trainable=train_func, resources={"cpu": 1, "gpu": 1}),
        param_space=get_search_space(args.model, args.dataset),
        run_config=run_configer,
        tune_config=tune_configer
    )

    results = tuner.fit()
    best_config = results.get_best_result(metric="val_f1", mode="max")
    df = results.get_dataframe()
    df.to_csv(f"out/{args.dataset}_{args.model}.csv")
    print(f"Best config: {best_config}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',"-m", type=str, required=True, help='model name', choices=_models)
    parser.add_argument('--dataset', "-d", type=str, required=True, help='dataset name', choices=_datasets)
    return parser.parse_args()
# %%
if __name__ == '__main__':
    args = parse_args()
    main(args)