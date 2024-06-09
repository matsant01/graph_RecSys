lr=0.0001
for hidden_channels in 128 256 512; do
    for num_conv_layers in 2 4; do
        for encoder_arch in SAGE GAT; do
            python scripts/trainer.py \
                --data_path ./data/splitted_data_small \
                --output_dir ./output \
                --num_conv_layers $num_conv_layers \
                --hidden_channels $hidden_channels \
                --num_decoder_layers 3 \
                --num_epochs 7500 \
                --validation_steps 500 \
                --encoder_arch $encoder_arch \
                --lr $lr \
                --loss mse \
                --device cuda:2
        done
    done
done