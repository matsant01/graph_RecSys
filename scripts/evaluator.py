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


def load_model(model_folder, full_data, config):
    if "encoder_arch" in config: 
        model = GNN(data=full_data,
                    conv_hidden_channels=config['hidden_channels'],
                    lin_hidden_channels=config['hidden_channels'],
                    num_conv_layers=config['num_conv_layers'],
                    num_decoder_layers=config['num_decoder_layers'],
                    use_embedding_layers=config['use_embedding_layers'], 
                    encoder_arch=config["encoder_arch"])
    else: 
        model = GNN(data=full_data,
                    conv_hidden_channels=config['hidden_channels'],
                    lin_hidden_channels=config['hidden_channels'],
                    num_conv_layers=config['num_conv_layers'],
                    num_decoder_layers=config['num_decoder_layers'],
                    use_embedding_layers=config['use_embedding_layers'])

    model_folder = os.path.join(model_folder, 'model.pt')
    model.load_state_dict(torch.load(model_folder))
    model.eval()
    return model

def load_data(test_data,  config):
    # using same loader as during training
    num_neighbors = [config['num_neighbors_in_sampling']] * config['num_iterations_loader']
    if config['sampler_type'] == "link-neighbor":
        test_loader =  LinkNeighborLoader(
            data=test_data,
            num_neighbors=num_neighbors,
            neg_sampling_ratio=0,  # no negative sampling, otherwise number of prediction wouldn't match the number of labels
            edge_label_index=(("user", "rates", "book"), test_data["user", "rates", "book"].edge_label_index),
            edge_label=test_data["user", "rates", "book"].edge_label,
            batch_size=config['batch_size'],
            shuffle=True,
        )

    elif config['sampler_type']  == "HGT":
        test_loader = HGTLoader(
            test_data,
            num_samples=num_neighbors,  
            shuffle=True,
            batch_size=config['batch_size'],
            input_nodes=("user", None),
        )
        
    return test_loader


def compute_save_metrics(test_data, predictions, model_folder):
    for k in [5, 10, 25]:
        test_data['predicted_rating'] = predictions
        top_k_recommendations = get_top_k_recommendations(test_data, k)

        # only consider item rated 4 or 5 as relevant
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
    
    test_data.to_csv(os.path.join(model_folder, 'predictions.csv'))

def evaluate_model_in_folder(model_folder, test_data, full_data, test_data_csv): 

    config = json.load(open(os.path.join(model_folder, 'config.json')))
    test_loader = load_data(test_data, config)

    model = load_model(model_folder, full_data, config).to(device)
    avg_loss, predictions = model.evaluation(test_loader, device)

    print(config)
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    compute_save_metrics(test_data_csv, predictions, model_folder)


def main(args, device):

    model_folder = args.model_folder
    data_folder = args.data_folder
    
    # check if model folder has modelpt file
    test_data = torch.load(os.path.join(data_folder, "test_hetero.pt")).to(device)
    test_data_csv = pd.read_csv(os.path.join(data_folder, "test.csv"))
    full_data = torch.load(os.path.join(data_folder, "data_hetero.pt")).to(device)

    print(os.path.join(model_folder, 'model.pt'))

    if os.path.exists(os.path.join(model_folder, 'model.pt')):
        evaluate_model_in_folder(model_folder, test_data, full_data, test_data_csv)

    else: 
        # iterate over all the model folders
        for model_dir in os.listdir(model_folder):
            model_dir = os.path.join(model_folder, model_dir)

            print(model_dir)
            # only consider folders with model.pt file
            if not os.path.exists(os.path.join(model_dir, 'model.pt')):
                continue

            evaluate_model_in_folder(model_dir, test_data, full_data, test_data_csv)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load models and make predictions.')
    parser.add_argument('--model_folder', type=str, required=True, help='Directory containing folder with model.pt files, a single folder with model.pt')
    parser.add_argument('--data_folder', type=str, required=True, help='Directory containing the data files.')
    
    args = parser.parse_args()

    # get device (evaluation isn't supported on mps)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    main(args, device)

    # Run the script
    # python scripts/evaluator.py --model_folder output --data_folder data/splitted_data
