# Aim of run: quantify impact of number of conv layers, hidden layers and hidden channels on performance
# Intuition: 4 conv layers should be enough, since it's a bipartite graph information should be local, 4 decoder layers should also 
# be sufficient

# Configuration:
# 1 - GNN, 2 conv layers, 64 hidden channels, 0 decoder layers, 10 epochs, lr=0.005, link-neighbor sampler
# 2 - GNN, 4 conv layers, 64 hidden channels, 4 decoder layers, 10 epochs, lr=0.005, link-neighbor sampler
# 3 - GNN, 4 conv layers, 128 hidden channels, 4 decoder layers, 10 epochs, lr=0.005, link-neighbor sampler

# Same configuration as matteo's notebook
python scripts/trainer.py \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --model_type GNN \
    --num_conv_layers 2 \
    --hidden_channels 64 \
    --num_decoder_layers 0 \
    --num_epochs 10 \
    --lr 0.005 \
    --sampler_type link-neighbor \
    --device cuda:1 \
    --verbose

python scripts/trainer.py \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --model_type GNN \
    --num_conv_layers 4 \
    --hidden_channels 64 \
    --num_decoder_layers 4 \
    --num_epochs 10 \
    --lr 0.005 \
    --sampler_type link-neighbor \
    --device cuda:1 \
    --verbose

python scripts/trainer.py \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --model_type GNN \
    --num_conv_layers 4 \
    --hidden_channels 128 \
    --num_decoder_layers 4 \
    --num_epochs 10 \
    --lr 0.005 \
    --sampler_type link-neighbor \
    --device cuda:1 \
    --verbose