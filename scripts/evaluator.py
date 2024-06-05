import os
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import LinkNeighborLoader, HGTLoader
import numpy as np
from src.models import GNN
import json

def load_model(model_folder, full_data, config):
    model = GNN(data=full_data,
                conv_hidden_channels=config['hidden_channels'],
                lin_hidden_channels=config['hidden_channels'],
                num_conv_layers=config['num_conv_layers'],
                num_decoder_layers=config['num_decoder_layers'])

    model_folder = os.path.join(model_folder, 'model.pt')
    model.load_state_dict(torch.load(model_folder))
    model.eval()
    return model

def load_data(test_data,  config):
    if config['sampler_type'] == "link-neighbor":
        test_loader =  LinkNeighborLoader(
            data=test_data,
            num_neighbors=[25, 25],
            neg_sampling_ratio=2,
            edge_label_index=(("user", "rates", "book"), test_data["user", "rates", "book"].edge_label_index),
            edge_label=test_data["user", "rates", "book"].edge_label,
            batch_size=4096,
            shuffle=True,
        )

    elif config['sampler_type']  == "HGT":
        test_loader = HGTLoader(
            test_data,
            num_samples=[1024] * 4,  
            shuffle=True,
            batch_size=128,
            input_nodes=("user", None),
        )
        
    return test_loader


def main(models_folder, data_folder):

    
    # check if model folder has modelpt file
    test_data = torch.load(os.path.join(data_folder, "test_hetero.pt"))
    full_data = torch.load(os.path.join(data_folder, "data_hetero.pt"))


    for model_folder in os.listdir(models_folder):
        model_folder = os.path.join(models_folder, model_folder)
        if not os.path.isdir(model_folder):
            continue
        if not os.path.exists(os.path.join(model_folder, 'model.pt')):
            continue
        
        config = json.load(open(os.path.join(model_folder, 'config.json')))
        test_loader = load_data(test_data, config)

        # TODO: write an eval function 
        model = load_model(model_folder, full_data, config)
        predictions = model(test_loader)
        print(predictions)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load models and make predictions.')
    parser.add_argument('--model_folder', type=str, required=True, help='Directory containing the model.pt file.')
    parser.add_argument('--data_folder', type=str, required=True, help='Directory containing the data files.')
    
    args = parser.parse_args()
    
    main(args.model_folder, args.data_folder)

    # python scripts/evaluator.py --model_folder output --data_folder data/splitted_data
