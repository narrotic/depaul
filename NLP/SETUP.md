# Local setup - CSC583 NLP (Fall 2026)

Course files copied from the instructor's repo:
<https://github.com/ntomuro/CSC583_2026Fall> (upstream history not kept here).

To pull later updates from the instructor, clone upstream somewhere temporary and copy
the new files over:

```bash
git clone https://github.com/ntomuro/CSC583_2026Fall.git /tmp/csc583
rsync -a --exclude='.git/' /tmp/csc583/ ~/workspace/depaul/NLP/
```

## Start working

This folder is a **uv project**, so there is nothing to activate:

```bash
cd ~/workspace/depaul/NLP
uv run jupyter lab
```

`uv run` resolves the environment first, so if the lockfile changed (or the `.venv` is
missing entirely) it installs what is needed before running. In a notebook, pick the
kernel **Python (CSC583 NLP)**.

Activating still works if you prefer it:

```bash
source .venv/bin/activate
jupyter lab
```

## Everyday commands

| Task | Command |
|---|---|
| Run anything | `uv run <cmd>` (e.g. `uv run python hw.py`) |
| Add a package | `uv add seaborn` |
| Remove a package | `uv remove seaborn` |
| Install what the lockfile says | `uv sync` |
| Upgrade everything | `uv lock --upgrade && uv sync` |
| Upgrade one package | `uv lock --upgrade-package transformers && uv sync` |
| See what is installed | `uv pip list` |

`uv add` edits `pyproject.toml` and `uv.lock` together, so the two never drift apart.

## The files

- `pyproject.toml` - the packages this course needs, grouped by assignment. Edit this (or
  use `uv add`) when something new is required.
- `uv.lock` - every package and exact version, resolved. Committed, so the environment is
  reproducible. Do not edit by hand.
- `requirements.txt` - generated from the lockfile for anywhere that cannot run uv, such
  as Google Colab. Regenerate after changing dependencies:
  ```bash
  uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
  ```
- `.python-version` - pins the interpreter to 3.13 so `uv run` always picks the same one.
- `.venv/` - the actual environment. Not committed; `uv sync` rebuilds it.

## The environment

- Python 3.13.8.
- PyTorch 2.14 with Apple **MPS** (GPU) support, verified working on this machine.
- 249 packages locked.

Rebuild from scratch on a new machine:

```bash
uv sync          # reads uv.lock, creates .venv, installs exact versions
```

Without `uv` at all:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## What's installed

| Area | Packages |
|---|---|
| Deep learning | `torch`, `torchvision`, `torchaudio` |
| Core data | `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn` |
| Notebooks | `jupyterlab`, `notebook`, `ipykernel`, `ipywidgets` |
| Classic NLP (HW1, HW4) | `nltk`, `gensim` |
| Transformers (HW5) | `transformers`, `tokenizers`, `accelerate`, `datasets`, `evaluate`, `sentencepiece`, `rouge_score`, `bert_score` |
| Embeddings & RAG (HW2, HW3) | `sentence-transformers`, `chromadb`, `langchain` (+ `-community`, `-chroma`, `-huggingface`, `-openai`, `-text-splitters`), `tiktoken`, `pypdf` |
| API clients | `openai`, `anthropic` |

NLTK corpora already downloaded to `~/nltk_data`: `punkt`, `punkt_tab`, `stopwords`,
`wordnet`, `omw-1.4`, `averaged_perceptron_tagger(_eng)`.

## Jupyter kernel

The kernel **Python (CSC583 NLP)** points at `.venv/bin/python`. If `uv sync` ever
recreates the environment somewhere else, re-register it:

```bash
uv run python -m ipykernel install --user --name csc583-nlp --display-name "Python (CSC583 NLP)"
```

## API keys

HW3 and HW5 call OpenAI / Anthropic. Put keys in `NLP/.env` (git-ignored at the repo root):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Load with `from dotenv import load_dotenv; load_dotenv()`. The notebooks are written for
Colab and use `google.colab.userdata` - swap those cells for `os.environ` when running
locally.

## Note on `.gitignore`

The repo-root `.gitignore` excludes `*.txt` and `*.zip`. `NLP/.gitignore` re-includes them
so the HW1 corpora and HW4 ngram token files are actually tracked.
