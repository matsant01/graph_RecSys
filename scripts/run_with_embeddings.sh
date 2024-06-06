#!/bin/bash -l
#SBATCH --chdir /scratch/izar/viel
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 90G
#SBATCH --time 15:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552


# New parameters to search:
# - With and without the use of learnable embeddings (HERE only with, check other scripts for without)
# - With more or less neighbors sampled by the loader

# Other paraters to search:
# - Number of conv layers
# - Number of decoder layers

# Ignore hyperparameters:
# - learning rate: we fix it to 0.005 (0.01 too high, 0.0001 too small)
# - hidden channels: we fix it to 128 (not too much difference between 64 and 128)

source ~/venvs/nml/bin/activate

lr=0.005
hidden_channels=128

# Number of conv layers
for conv_layers in 2 4; do
    # Number of decoder layers        
    for num_neighbors in 100 200 1000; do
        python /scratch/izar/viel/graph_RecSys/scripts/trainer.py \
        --data_path /scratch/izar/viel/graph_RecSys/data/splitted_data \
        --output_dir /scratch/izar/viel/graph_RecSys/output \
        --model_type GNN \
        --num_conv_layers $conv_layers \
        --hidden_channels $hidden_channels \
        --num_decoder_layers 4 \
        --num_epochs 15 \
        --lr $lr \
        --sampler_type link-neighbor \
        --num_neighbors $num_neighbors \
        --batch_size 4096 \
        --device cuda \
        --use_embedding_layers \
        --num_iterations_loader 2 \
        --verbose
    done
done