# Aim of run: quantify impact of learning rate on performance

# Configuration:
# 1 - GNN, 4 conv layers, 64 hidden channels, 4 decoder layers, 10 epochs, lr=0.001, link-neighbor sampler
# 2 - GNN, 4 conv layers, 64 hidden channels, 4 decoder layers, 10 epochs, lr=0.0001, link-neighbor sampler
# 3 - GNN, 4 conv layers, 64 hidden channels, 4 decoder layers, 10 epochs, lr=0.01, link-neighbor sampler


python scripts/trainer.py \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --model_type GNN \
    --num_conv_layers 4 \
    --hidden_channels 64 \
    --num_decoder_layers 4 \
    --num_epochs 10 \
    --lr 0.001 \
    --sampler_type link-neighbor \
    --device cuda:2 \
    --verbose

python scripts/trainer.py \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --model_type GNN \
    --num_conv_layers 4 \
    --hidden_channels 64 \
    --num_decoder_layers 4 \
    --num_epochs 10 \
    --lr 0.0001 \
    --sampler_type link-neighbor \
    --device cuda:2 \
    --verbose

python scripts/trainer.py \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --model_type GNN \
    --num_conv_layers 4 \
    --hidden_channels 64 \
    --num_decoder_layers 4 \
    --num_epochs 10 \
    --lr 0.01 \
    --sampler_type link-neighbor \
    --device cuda:2 \
    --verbose