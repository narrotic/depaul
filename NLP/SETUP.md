# Local setup — CSC583 NLP (Fall 2026)

Course files copied from the instructor's repo:
<https://github.com/ntomuro/CSC583_2026Fall> (upstream history not kept here).

To pull later updates from the instructor, clone upstream somewhere temporary and copy
the new files over:

```bash
git clone https://github.com/ntomuro/CSC583_2026Fall.git /tmp/csc583
rsync -a --exclude='.git/' /tmp/csc583/ ~/workspace/depaul/NLP/
```

## Start working

```bash
cd ~/workspace/depaul/NLP
source .venv/bin/activate
jupyter lab            # or just start coding
```

In a notebook, pick the kernel **Python (CSC583 NLP)**.

## The environment

- Python 3.13.8, virtual env at `.venv/` (not committed).
- PyTorch 2.14 with Apple **MPS** (GPU) support — verified working on this machine.
- `requirements.in` = the top-level packages we actually care about.
  `requirements.txt` = the full pinned freeze (what to reinstall from).

Rebuild from scratch:

```bash
uv venv --python 3.13 .venv
uv pip install -r requirements.txt     # exact pins
# or: uv pip install -r requirements.in # latest compatible versions
```

Without `uv`:

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

## API keys

HW3 and HW5 call OpenAI / Anthropic. Put keys in `NLP/.env` (git-ignored at the repo root):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Load with `from dotenv import load_dotenv; load_dotenv()`. The notebooks are written for
Colab and use `google.colab.userdata` — swap those cells for `os.environ` when running locally.

## Note on `.gitignore`

The repo-root `.gitignore` excludes `*.txt` and `*.zip`. `NLP/.gitignore` re-includes them
so the HW1 corpora and HW4 ngram token files are actually tracked.
