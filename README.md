# irg_final_project
Final project for the Information Retrieval and Generation class

# Getting Started
## 1. Download the repository
```
git clone https://github.com/olaghattas/irg_final_project.git
cd irg_final_project
```

## 2. Install dependencies
1) (Recommended) create/activate a conda environment:
```
conda create -n rag python=3.11
conda activate rag
```
2) Install Python packages:
```
pip install -r requirements.txt
```

## 3. Download dataset
```
cd src/getting_started
python download_dataset.py
```

<br> <br>

# Project Layout

irg_final_project/
├── lib/
│   ├── Document.py
│   ├── Embedding.py
│   ├── Index.py
│   ├── Query.py
│   ├── Result.py
│   └── __init__.py
│
├── dataset/
│   ├── LitSearch_corpus_clean/
│   │   ├── dataset_info.json
│   │   ├── data-00000-of-00001.arrow
│   │   └── state.json
│   └── LitSearch_query/
│       ├── dataset_info.json
│       └── data-00000-of-00001.arrow
│
├── run_files/
│   └── bm25.run
│
└── pyserini_index/
