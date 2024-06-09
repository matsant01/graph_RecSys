import os
import sys
import json
import argparse
import numpy as np

import torch
from torch_geometric.loader import LinkNeighborLoader, HGTLoader

sys.path.append("./.")

from src.models import GNN
from src.evaluation_metrics import * 


def load_model(model_folder, full_data, config, evaluate_last=False):
    if "encoder_arch" in config: 
        model = GNN(
            data=full_data,
            conv_hidden_channels=config['hidden_channels'],
            lin_hidden_channels=config['hidden_channels'],
            num_conv_layers=config['num_conv_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            use_embedding_layers=config['use_embedding_layers'],
            encoder_arch=config['encoder_arch'],
            out_dim=1 if config["loss"] != "nll" else 5,   
        )
    else: 
        model = GNN(data=full_data,
                    conv_hidden_channels=config['hidden_channels'],
                    lin_hidden_channels=config['hidden_channels'],
                    num_conv_layers=config['num_conv_layers'],
                    num_decoder_layers=config['num_decoder_layers'],
                    use_embedding_layers=config['use_embedding_layers'])

    model_folder = os.path.join(model_folder, 'model.pt' if evaluate_last else 'best_model.pt')
    model.load_state_dict(torch.load(model_folder))
    model.eval()
    return model


def evaluate_model_in_folder(model_folder, test_data, full_data, evaluate_last=False):

    config = json.load(open(os.path.join(model_folder, 'config.json')))

    model = load_model(model_folder, full_data, config, evaluate_last=evaluate_last).to(device)
    _, metrics = model.evaluation_full_batch(val_data=test_data, device=device, criterion=None, k=5, big_k=15)

    print(config)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value}")
    print("\n\n")
    
    with open(os.path.join(model_folder, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    

def main(args, device):

    model_folder = args.model_folder
    file_name = "model.pt" if args.evaluate_last else "best_model.pt"
    data_folder = args.data_folder
    
    # check if model folder has modelpt file
    test_data = torch.load(os.path.join(data_folder, "test_hetero.pt")).to(device)
    test_data_csv = pd.read_csv(os.path.join(data_folder, "test.csv"))
    full_data = torch.load(os.path.join(data_folder, "data_hetero.pt")).to(device)
    model_path = os.path.join(model_folder, file_name)
    print(model_path)

    if os.path.exists(model_path):
        evaluate_model_in_folder(model_folder, test_data, full_data, test_data_csv, evaluate_last=args.evaluate_last)
    else: 
        # iterate over all the model folders
        for model_dir in os.listdir(model_folder):
            model_dir = os.path.join(model_folder, model_dir)

            print(model_dir)
            # only consider folders with model.pt file
            if not os.path.exists(os.path.join(model_dir, file_name)):
                continue

            evaluate_model_in_folder(model_dir, test_data, full_data, evaluate_last=args.evaluate_last)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load models and make predictions.')
    parser.add_argument('--model_folder', type=str, required=True, help='Directory containing folder with model.pt files, a single folder with model.pt')
    parser.add_argument('--data_folder', type=str, required=True, help='Directory containing the data files.')
    parser.add_argument('--evaluate_last', action='store_true', help='Evaluate the last model in the model folder instead of the best.')
    
    args = parser.parse_args()

    # get device (evaluation isn't supported on mps)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    main(args, device)

    # Run the script
    # python scripts/evaluator.py --model_folder output --data_folder data/splitted_data
