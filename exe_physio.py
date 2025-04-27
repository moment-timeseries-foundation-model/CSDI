import argparse
import datetime
import json
import yaml
import os

from main_model import CSDIModelPhysio
from dataset_physio import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument(
    "--config",
    type=str,
    default="base.yaml",
    help="Path to the configuration file (YAML format).",
)
parser.add_argument(
    "--device",
    default="cuda:0",
    help="Device to use for computation (e.g., 'cuda:0' or 'cpu').",
)
parser.add_argument(
    "--seed", type=int, default=1, help="Random seed for reproducibility."
)
parser.add_argument(
    "--testmissingratio",
    type=float,
    default=0.1,
    help="Ratio of missing data during testing.",
)
parser.add_argument(
    "--nfold",
    type=int,
    default=0,
    help="Fold index for 5-fold cross-validation (valid values: [0-4]).",
)
parser.add_argument(
    "--unconditional",
    action="store_true",
    help="Flag to indicate if the model is unconditional.",
)
parser.add_argument(
    "--nsample",
    type=int,
    default=100,
    help="Number of samples to generate during evaluation.",
)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/physio_fold" + str(args.nfold) + "_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = CSDIModelPhysio(config, args.device).to(args.device)

# Train the model
train(
    model,
    config["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=foldername,
)

# Evaluate the model on the test set
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
