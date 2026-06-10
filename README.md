# NLP Tandem / Jointe

This project compares text representation and clustering approaches on three
labeled datasets: Classic 3, Classic 4, and BBC.

The repository contains two main experiment families:

- **Tandem approach**: train text embeddings first, then apply dimensionality
  reduction methods such as PCA, t-SNE, UMAP, and autoencoders before
  clustering with K-means++, K-medoids, spherical K-means, and hierarchical
  clustering.
- **Joint / simultaneous approach**: apply methods that combine dimensionality
  reduction and clustering, such as Reduced K-means and Deep K-means.

## Repository Structure

```text
content/                  Input datasets
datascience2_w2vect.ipynb Word2Vec notebook experiments
datascience2_Glove.ipynb  GloVe notebook experiments
R/                        Reduced K-means scripts for GloVe embeddings
w2v_R/                    Reduced K-means scripts for Word2Vec embeddings
scripts/                  Reusable setup and training scripts
```

## Python Setup

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use Windows PowerShell, activate the virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## R Setup

The Reduced K-means scripts use the R package `clustrd`.

```bash
Rscript scripts/install_r_dependencies.R
```

## Run The Notebooks

Start Jupyter after installing the Python requirements:

```bash
jupyter notebook
```

Then open one of these notebooks:

- `datascience2_w2vect.ipynb`
- `datascience2_Glove.ipynb`

The notebooks expect the CSV files in `content/` and produce embedding CSVs used
by the R clustering scripts.

## Train Word2Vec Embeddings From The Command Line

The notebooks are useful for exploration, but the reusable script is easier to
rerun and version control.

Classic 3:

```bash
python scripts/train_word2vec_embeddings.py \
  --input content/classic3.csv \
  --output w2v_R/df1_embeddings_w2v.csv \
  --vector-size 300 \
  --label-column label \
  --text-column text
```

Classic 4:

```bash
python scripts/train_word2vec_embeddings.py \
  --input content/classic4.csv \
  --output w2v_R/df2_embeddings_w2v.csv \
  --vector-size 300 \
  --label-column label \
  --text-column text
```

BBC:

```bash
python scripts/train_word2vec_embeddings.py \
  --input content/bbc.csv \
  --output w2v_R/df3_embeddings_w2v.csv \
  --vector-size 300 \
  --label-column label \
  --text-column text
```

The output format is compatible with the existing R scripts: embedding columns
named `0`, `1`, ..., `vector_size - 1`, followed by the original `label`.

## Run Reduced K-means

Run the R scripts from the folder that contains the matching embeddings:

```bash
cd w2v_R
Rscript "reduced_kmeans -df1.R"
Rscript "reduced_kmeans -df2.R"
Rscript "reduced_kmeans-df3.R"
```

For GloVe embeddings:

```bash
cd R
Rscript "reduced_kmeans -df1.R"
Rscript "reduced_kmeans -df2.R"
Rscript "reduced_kmeans-df3.R"
```

## Notes

- GloVe experiments require a local GloVe text file. The large pre-trained
  GloVe files are not stored in this repository.
- The dependency versions in `requirements.txt` avoid common notebook import
  issues, including the `gensim` / `scipy` compatibility problem.
- Generated embedding CSV files can be large. Commit regenerated outputs only
  when they are needed for reproducible project results.
