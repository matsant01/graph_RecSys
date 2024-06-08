num_conv_layers=2
encoder_arch=SAGE

for hidden_channels in 128 256 512; do
    for lr in 0.01 0.001 0.0001; do
        python scripts/trainer.py \
            --data_path ./data/splitted_data \
            --output_dir ./output \
            --num_conv_layers $num_conv_layers \
            --hidden_channels $hidden_channels \
            --num_decoder_layers 3 \
            --num_epochs 7500 \
            --validation_steps 750 \
            --encoder_arch $encoder_arch \
            --lr $lr \
            --loss mse \
            --device cuda
    done
done