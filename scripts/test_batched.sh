num_conv_layers=2
encoder_arch=SAGE
hidden_channels=256
lr=0.00025

python scripts/trainer.py \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --num_conv_layers $num_conv_layers \
    --hidden_channels $hidden_channels \
    --num_decoder_layers 3 \
    --sampler_type link-neighbor \
    --num_epochs 10 \
    --batch_size 4096 \
    --encoder_arch $encoder_arch \
    --lr $lr \
    --loss mse \
    --device cuda:0 \
    --verbose