import argparse
import os
import pandas as pd
from datetime import datetime
from src.evaluation_metrics import *
from src.matrix_factorization import MatrixFactorization 

def main():
    parser = argparse.ArgumentParser(description="Train a Matrix Factorization model.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing the split data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the output logs will be saved.')
    parser.add_argument('--num_factors', type=int, default=5, help='Number of latent factors.')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--lambda_reg', type=float, default=0.01, help='Regularization parameter.')
    parser.add_argument('--log_every', type=int, default=100, help='Frequency of batches to log the loss.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')

    args = parser.parse_args()

    # Ensure the output directory exists
    root_output_dir = args.output_dir
    output_dir = os.path.join(root_output_dir, f"matrixFact_lr={args.learning_rate}_reg={args.lambda_reg}_epochs={args.num_epochs}_factors={args.num_factors}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    train_data = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    eval_data = pd.read_csv(os.path.join(args.data_path, 'val.csv'))
    test_data = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
    full_data = pd.concat([train_data, eval_data])

    # Initialize the model
    model = MatrixFactorization(
        num_factors=args.num_factors,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_reg=args.lambda_reg,
        log_every=args.log_every,
        output_dir=output_dir
    )

    # Train the model
    model.train_loop(full_data, train_data, eval_data)

    k = 5
    test_data['predicted_rating'] = model.predict(test_data)
    top_k_recommendations = get_top_k_recommendations(test_data, k)
    actual_items = get_actual_items(test_data, 4) # ground truth
    
    print("--- Evaluation ---")

    # Evaluate the recommendations  
    mean_precision, mean_recall, mean_f1 = evaluate_recommendations(top_k_recommendations, actual_items, k)
    print(f"Mean Precision@{k}: {mean_precision}")
    print(f"Mean Recall@{k}: {mean_recall}")
    print(f"Mean F1 Score@{k}: {mean_f1}")

    # save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Mean Precision@{k}: {mean_precision}\n")
        f.write(f"Mean Recall@{k}: {mean_recall}\n")
        f.write(f"Mean F1 Score@{k}: {mean_f1}\n")

if __name__ == '__main__':
    main()
