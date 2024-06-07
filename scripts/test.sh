python scripts/trainer.py \
    --data_path ./data/splitted_data_small \
    --output_dir ./output \
    --num_conv_layers 4 \
    --hidden_channels 128 \
    --num_decoder_layers 3 \
    --num_epochs 15 \
    --lr 0.005 \
    --loss mse \
    --device mps \
    --verbose