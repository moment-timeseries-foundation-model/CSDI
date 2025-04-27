import argparse
import datetime
import json
import yaml
import os

from dataset_pm25 import get_dataloader
from main_model import CSDIModelPM25
from utils import train, evaluate

parser = argparse.ArgumentParser(
    description="CSDI - Conditional Score-based Diffusion Model"
)
parser.add_argument(
    "--config",
    type=str,
    default="base.yaml",
    help="Path to the configuration file (default: base.yaml)",
)
parser.add_argument(
    "--device",
    default="cuda:0",
    help="Device to run the model on (e.g., 'cuda:0' for GPU or 'cpu' for CPU)",
)
parser.add_argument(
    "--targetstrategy",
    type=str,
    default="mix",
    choices=["mix", "random"],
    help="Strategy for selecting target data (choices: 'mix', 'random')",
)
parser.add_argument(
    "--validationindex",
    type=int,
    default=0,
    help="Index of the month used for validation (value range: [0-7])",
)
parser.add_argument(
    "--nsample",
    type=int,
    default=100,
    help="Number of samples to generate during evaluation",
)
parser.add_argument(
    "--unconditional",
    action="store_true",
    help="Flag to train the model in an unconditional setting",
)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = (
    "./save/pm25_validationindex" + str(args.validationindex) + "_" + current_time + "/"
)

print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    config["train"]["batch_size"], device=args.device, validindex=args.validationindex
)
model = CSDIModelPM25(config, args.device).to(args.device)

# Train the model
train(
    model,
    config["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=foldername,
)

# Evaluate the model
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
