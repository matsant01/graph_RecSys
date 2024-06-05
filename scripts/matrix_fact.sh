python -m scripts.matrix_fact_trainer \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --learning_rate 0.01 \
    --num_epochs 3 \
    --num_factors 10 \

python -m scripts.matrix_fact_trainer \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --learning_rate 0.05 \
    --num_epochs 3 \
    --num_factors 10 \

python -m scripts.matrix_fact_trainer \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --learning_rate 0.001 \
    --num_epochs 5 \
    --num_factors 10 \

python -m scripts.matrix_fact_trainer \
    --data_path ./data/splitted_data \
    --output_dir ./output \
    --learning_rate 0.05 \
    --num_epochs 5 \
    --num_factors 100 \