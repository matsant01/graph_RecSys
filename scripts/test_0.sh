for lr in 0.0005 0.001 0.0001 0.00005; do
    for hidden_channels in 128 256 512; do
        for num_conv_layers in 2 4; do
            python scripts/trainer.py \
                --data_path ./data/splitted_data_small \
                --output_dir ./output \
                --num_conv_layers $num_conv_layers \
                --hidden_channels $hidden_channels \
                --num_decoder_layers 3 \
                --num_epochs 10000 \
                --encoder_arch SAGE \
                --lr $lr \
                --loss mse \
                --device cuda:0 \
                --verbose
        done
    done
done