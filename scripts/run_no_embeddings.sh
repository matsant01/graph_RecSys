# New parameters to search:
# - With and without the use of learnable embeddings (HERE only without, check other scripts for with)
# - With more or less neighbors sampled by the loader

# Other paraters to search:
# - Number of conv layers
# - Number of decoder layers

# Ignore hyperparameters:
# - learning rate: we fix it to 0.005 (0.01 too high, 0.0001 too small)
# - hidden channels: we fix it to 128 (not too much difference between 64 and 128)


lr=0.005
hidden_channels=128

# Number of conv layers
for conv_layers in 2 4; do
    # Number of decoder layers
    for decoder_layers in 0 2; do
        
        for num_neighbors in 25 100 500 1000; do
            python scripts/trainer.py \
            --data_path ./data/splitted_data \
            --output_dir ./output \
            --model_type GNN \
            --num_conv_layers $conv_layers \
            --hidden_channels $hidden_channels \
            --num_decoder_layers $decoder_layers \
            --num_epochs 15 \
            --lr $lr \
            --sampler_type link-neighbor \
            --num_neighbors $num_neighbors \
            --batch_size 4096 \
            --device cuda:1 \
            --num_iterations_loader 2 \
            --verbose
            # NOTE: NOT USING LEARNABLE EMBEDDINGS
        done
    done
done