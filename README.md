# Heterogeneous Graph Convolution for Book Recommendations
Final Project for Network Machine Lerarning course at EPFL (EE-452)
**Authors**:
- [Matteo Santelmo](https://github.com/matsant01) - SCIPER: 376844
- [Stefano Viel](https://github.com/stefanoviel) - SCIPER: 377251

---

## Repository structure
The repository is structured as follows:
- `data/`: contains the original dataset used for the project and is used to store the processed data.
- `notebooks/`: contains the Jupyter notebooks used for the project:
    - [`data_exploration.ipynb`](notebooks/data_exploration.ipynb) provides some insights on the dataset.
    - [`baselines.ipynb`](notebooks/baselines.ipynb) contains the code used to train and evaluate the two baselines.
    - [`results_analysis.ipynb`](notebooks/results_analysis.ipynb) contains the code used to analyze the results of the models obtained via grid search and the final experiments.
- `src/`: contains the source code of the project, in particular:
    - [`models.py`](src/models.py) contains the implementation of the GCN-based Encoder-Decoder architecture used for the project.
    - [`evaluation_metrics.py`](src/evaluation_metrics.py) contains the implementation of the evaluation metrics.
    - [`matrix_factorization.py`](src/matrix_factorization.py) implements the Matrix Factorization baseline.
- `scripts/`: contains the scripts used to run the experiments.
- `report/`: contains the final report of the project.

# Running the code
### Getting started
First of all you need to install the required packages. We recommend to create a virtual environment an install the packages there. You can do so by running the following commands:
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Scripts usage 
Now you can run all the code by using the scripts provided in the `scripts/` folder. By running any python script with the `--help` flag you can see the available options.

To create and store both the Heterogeneous Graph and the training-validation-test splits you can use:
```shell
mkdir -p ./data/splitted_data
python scripts/create_datasets.py --save_dir ./data/splitted_data
# by adding the --add_extra_data option, the graph will also contain authors and language nodes
```

To train a model you can run [`scripts/trainer.py`](scripts/trainer.py) with appropriate arguments. This would automatically create a folder in the specified output directory containing the model file (both the last and the best), the TensorBoard logs and a configuration file with the hyperparameters used. For example:
```shell
python scripts/trainer.py \
--data_path ./data/splitted_data \
--output_dir ./output \
--num_conv_layers 2 \
--hidden_channels 256 \
--num_decoder_layers 3\
--sampler_type link-neighbor \
--num_epochs 10 \
--batch_size 1024 \
--encoder_arch SAGE \
--validation_steps -1 \
--lr 0.00025 \
--loss mse \
--device cuda:0 \
--verbose
```
Finally, to evaluate your models you can use [`scripts/evaluator.py`](scripts/evaluator.py) with appropriate arguments depending on where your model and data are stored. This script will create a `metrics.json` file in the model folder containing the values for the evaluation metrics.
```shell
python scripts/evaluator.py \
--model_folder ./output \
--data_folder ./data/splitted_data \
# adding --evaluate_last the evaluator will consider the last model instead of the best one
```
In this [`example.sh`](scripts/shell/example.sh) you can find a script that runs the whole pipeline with some default parameters and different models.

## Results
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow">Model</th>
    <th class="tg-c3ow">MAP@15</th>
    <th class="tg-c3ow">Precision@5</th>
    <th class="tg-c3ow">Recall@5</th>
    <th class="tg-c3ow">F1@5</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-abip">Random Baseline</td>
    <td class="tg-abip">0.471</td>
    <td class="tg-abip">0.472</td>
    <td class="tg-abip">0.332</td>
    <td class="tg-abip">0.379</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Matrix Factorization</td>
    <td class="tg-c3ow">0.489</td>
    <td class="tg-c3ow">0.494</td>
    <td class="tg-c3ow">0.312</td>
    <td class="tg-c3ow">0.371</td>
  </tr>
  <tr>
    <td class="tg-abip">EncDec with SAGE</td>
    <td class="tg-abip">0.551</td>
    <td class="tg-abip">0.552</td>
    <td class="tg-abip">0.347</td>
    <td class="tg-abip">0.414</td>
  </tr>
  <tr>
    <td class="tg-c3ow">EncDec with SAGE<br>+ Additional Nodes</td>
    <td class="tg-c3ow"><span style="font-weight:bold">0.593</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">0.596</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">0.380</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">0.450</span></td>
  </tr>
</tbody></table>
