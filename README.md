
# Environment Setup

## Conda Installation
To install Conda (if you have not), follow these steps (See [Miniconda Documentation](https://docs.anaconda.com/miniconda/)):

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

### Initialize Conda
```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

Create from a yml file (recommended):
```bash
conda env create -f environment.yml
```

## Jupyter Notebook Kernel Setup
This repo involves a Jupyter Notebook for results visualization. In case that The Conda virtual environment may not show up in the Jupyter Notebook, you may follow the steps below to resolve this(See [Stack Overflow](https://stackoverflow.com/questions/41518093/cant-find-jupyter-notebook-kernel?newreg=5d67af917ea8482f8fc944a65c42a058)):

```bash
<!-- conda create -n myenv python=3.12.0 -->
conda activate myenv
conda install -y -c conda-forge jupyterlab
conda install -y -c anaconda jupyter
conda install -y ipykernel
conda install nb_conda
python -m ipykernel install --user --name myenv
```



## Shapr Implementation (for Conditional Imputation methods)
To set up the `shapr` implementation (See [Shapr Repository](https://github.com/NorskRegnesentral/shapr/tree/master)):

```bash
cd ~
git clone https://github.com/NorskRegnesentral/shapr/tree/master
```

Follow the instructions here: [Shapr Setup](https://github.com/NorskRegnesentral/shapr/tree/master/python)

---

# Running the Scripts

1. **Run Training, Explanation, and Post-Processing**:

    Inside the home folder (`./pairwise_shapley/`), go to the `./pairwise_shapley/model` folder. Run the provided bash script (`train_exp_post.sh`) which supports `MODEL_VERSION` variables v0, v1, v2, v3. You can specify the model version in the command line. This bash file will run training, explanation (for all specified methods), and post-process the explanation results in one go.

    ```bash
    cd ~/pairwise_shapley/model
    ./train_exp_post.sh 10
    ```

    Ensure to replace `10` with the desired `MODEL_VERSION`. This script will train the model, perform explanations using various methods, and post-process the results (e.g., normalize Shapley values for different methods, preparing them for better comparison).

2. **Visualization**:

    After completing step 1, you can visualize the final results using the Jupyter Notebook located at `./pairwise_shapley/viz_final.ipynb`.

    Simply click the notebook in Jupyter or :

    ```bash
    jupyter notebook ./pairwise_shapley/viz_final.ipynb
    ```

    This notebook will help you reproduce the same visualizations seen in the presentation deck.
