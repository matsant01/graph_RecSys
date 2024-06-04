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