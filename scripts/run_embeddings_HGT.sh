# New parameters to search:
# - With and without the use of learnable embeddings (HERE only with, check other scripts for without)
# - With more or less neighbors sampled by the loader

# Other paraters to search:
# - Number of conv layers
# - Number of decoder layers

# Ignore hyperparameters:
# - learning rate: we fix it to 0.005 (0.01 too high, 0.0001 too small)
# - hidden channels: we fix it to 128 (not too much difference between 64 and 128)

lr=0.005
hidden_channels=128

python scripts/trainer.py \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --model_type GNN \
    --num_conv_layers 4 \
    --hidden_channels 128 \
    --num_decoder_layers 4 \
    --num_epochs 15 \
    --lr $lr \
    --sampler_type HGT \
    --num_neighbors 1 \
    --batch_size 4096 \
    --device cpu \
    --use_embedding_layers \
    --num_iterations_loader 2 \
    --verbose
