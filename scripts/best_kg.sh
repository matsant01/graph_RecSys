num_conv_layers=4
encoder_arch=SAGE
hidden_channels=256
lr=0.00025

python scripts/trainer.py \
    --data_path ./data/splitted_knowledge_graph \
    --output_dir ./output_kg \
    --num_conv_layers $num_conv_layers \
    --hidden_channels $hidden_channels \
    --num_decoder_layers 3\
    --sampler_type link-neighbor \
    --num_epochs 50 \
    --batch_size 4096 \
    --encoder_arch $encoder_arch \
    --validation_steps -1 \
    --lr $lr \
    --loss mse \
    --device cuda \
    --verbose