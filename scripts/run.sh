# # Create datasets
# echo "Creating datasets"
# mkdir -p ./data/splitted_data_kg
# python src/load_split_data.py --save_dir ./data/splitted_data_kg --add_extra_data

# mkdir -p ./data/splitted_data
# python src/load_split_data.py --save_dir ./data/splitted_data

# echo "Training models"

# # Start training with users and books only
# num_conv_layers=2
# encoder_arch=SAGE
# hidden_channels=256
# lr=0.00025

# python scripts/trainer.py \
#     --data_path ./data/splitted_data \
#     --output_dir ./output \
#     --num_conv_layers $num_conv_layers \
#     --hidden_channels $hidden_channels \
#     --num_decoder_layers 3\
#     --sampler_type link-neighbor \
#     --num_epochs 30 \
#     --batch_size 8192 \
#     --encoder_arch $encoder_arch \
#     --validation_steps -2 \
#     --lr $lr \
#     --loss mse \
#     --device cuda \
#     --verbose


# # Start training with kwoledge graph
# num_conv_layers=4
# encoder_arch=SAGE
# hidden_channels=256
# lr=0.00025

# python scripts/trainer.py \
#     --data_path ./data/splitted_data_kg \
#     --output_dir ./output_kg \
#     --num_conv_layers $num_conv_layers \
#     --hidden_channels $hidden_channels \
#     --num_decoder_layers 3\
#     --sampler_type link-neighbor \
#     --num_epochs 30 \
#     --batch_size 8192 \
#     --encoder_arch $encoder_arch \
#     --validation_steps -2 \
#     --lr $lr \
#     --loss mse \
#     --device cuda \
#     --verbose



# Evaluation
echo "Evaluating models"

python scripts/evaluator.py \
    --model_folder ./output \
    --data_folder ./data/splitted_data

python scripts/evaluator.py \
    --model_folder ./output \
    --data_folder ./data/splitted_data \
    --evaluate_last

python scripts/evaluator.py \
    --model_folder ./output_kg \
    --data_folder ./data/splitted_data_kg

python scripts/evaluator.py \
    --model_folder ./output_kg \
    --data_folder ./data/splitted_data_kg \
    --evaluate_last