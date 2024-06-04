# Aim of run: does sampler type impact performance?

# Intuition: maybe a better sample can allow a higher learning rate? 

# Configuration:
# 1 - GNN, 4 conv layers, 64 hidden channels, 4 decoder layers, 10 epochs, lr=0.005, HGT sampler
# 2 - GNN, 4 conv layers, 128 hidden channels, 4 decoder layers, 10 epochs, lr=0.01, HGT sampler

python scripts/trainer.py \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --model_type GNN \
    --num_conv_layers 4 \
    --hidden_channels 64 \
    --num_decoder_layers 4 \
    --num_epochs 10 \
    --lr 0.005 \
    --sampler_type HGT \
    --device cuda:3 \
    --verbose

python scripts/trainer.py \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --model_type GNN \
    --num_conv_layers 4 \
    --hidden_channels 128 \
    --num_decoder_layers 4 \
    --num_epochs 10 \
    --lr 0.01 \
    --sampler_type HGT \
    --device cuda:3 \
    --verbose
