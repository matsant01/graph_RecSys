import os
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import LinkNeighborLoader, HGTLoader
from src.evaluation_metrics import * 
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
            neg_sampling_ratio=0,  # no negative sampling, otherwise number of prediction wouldn't match the number of labels
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


def compute_metrics(test_data, predictions, model_folder):
    k = 5
    test_data['predicted_rating'] = predictions
    top_k_recommendations = get_top_k_recommendations(test_data, k)
    actual_items = get_actual_items(test_data, 4) # ground truth

    # Evaluate the recommendations
    mean_precision, mean_recall, mean_f1 = evaluate_recommendations(top_k_recommendations, actual_items, k)
    print(f"Mean Precision@{k}: {mean_precision}")
    print(f"Mean Recall@{k}: {mean_recall}")
    print(f"Mean F1 Score@{k}: {mean_f1}")

    metrics = {
        f"Mean Precision@{k}": mean_precision,
        f"Mean Recall@{k}": mean_recall,
        f"Mean F1 Score@{k}": mean_f1
    }

    with open(os.path.join(model_folder, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def main(args, device):

    models_folder = args.model_folder
    data_folder = args.data_folder
    
    # check if model folder has modelpt file
    test_data = torch.load(os.path.join(data_folder, "test_hetero.pt")).to(device)
    test_data_csv = pd.read_csv(os.path.join(data_folder, "test.csv"))
    full_data = torch.load(os.path.join(data_folder, "data_hetero.pt")).to(device)

    for model_folder in os.listdir(models_folder):
        model_folder = os.path.join(models_folder, model_folder)
        if not os.path.isdir(model_folder):
            continue
        if not os.path.exists(os.path.join(model_folder, 'model.pt')):
            continue
        
        config = json.load(open(os.path.join(model_folder, 'config.json')))
        test_loader = load_data(test_data, config)

        model = load_model(model_folder, full_data, config).to(device)
        avg_loss, predictions = model.evaluation(test_loader, device)

        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        # labels = torch.cat(labels, dim=0).cpu().numpy()
        compute_metrics(test_data_csv, predictions, model_folder)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load models and make predictions.')
    parser.add_argument('--model_folder', type=str, required=True, help='Directory containing the model.pt file.')
    parser.add_argument('--data_folder', type=str, required=True, help='Directory containing the data files.')
    
    args = parser.parse_args()

    # get device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    main(args, device)

    # python scripts/evaluator.py --model_folder output --data_folder data/splitted_data
