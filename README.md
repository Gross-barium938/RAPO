# Codes for KDD 2026 Submission 1157 "RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization"

We sincerely thank the reviewers for their time and contributions during the review process.


This repository provides the official implementation of RAPO, a retrieval-augmented Agentic RL framework designed to expand policy exploration for LLM agents.

## Quick Start

### Step 1: Environment Installation

We recommend creating separate environments for RL training and retrieval services to avoid dependency conflicts.

#### RL Environment
```bash
conda create -n rapo python=3.12.9
conda activate rapo
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0  
pip install vllm==0.8.5.post1
# verl
pip install -e .
# flash attention 2
pip install flash-attn --no-build-isolation
pip install swanlab
```

#### Search & Retrieval Environment
```bash
conda create -n retriever python=3.10.13
conda activate retriever
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0  
pip install transformers datasets pyserini
# install the gpu version faiss to guarantee efficient RL rollout
pip install faiss-gpu==1.7.3
# API function
pip install uvicorn fastapi
```


### Step 2: Search Server & Python Server

Download the Wikipedia corpus from: 👉 [here](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus). Save the file under `./wiki` and decompress it:

```bash
gzip -dk $.wiki/wiki-18.jsonl.gz
```

Then build and launch the search server:

```bash
conda activate retriever
python  build_search_index.py
python  search_server.py
```

To buld python server, specify your Conda path and environment name in both configuration files: `./Buffer/scripts/config/ppo_trainer.yaml` and `./RAPO/scripts/config/ppo_trainer.yaml`.

Update the following section:

```bash
tool_instances:
	python:  
	  class_path: verl.workers.rollout.tools.python_tool.PythonTool
	  params:  
	    conda_path: [your_conda_path]
	    conda_env: [your_python_env_name]
```
Ensure that the specified environment is correctly installed and accessible.

### Step 3: Step-Trace Buffer Construction
Next, construct the Step-Trace Buffer using AEPO-Qwen3-14B to support retrieval-augmented rollout. Run:

```bash
conda activate rapo
bash  ./Buffer/scripts/Reasoning_corpus.sh
```



### Step 4: Retrieval Server

Similar to the search module, the retrieval component is deployed as a standalone service. Run:

```bash
conda activate retriever
python  build_retrieve_index.py
python  retrieve_server.py
```


### Step 5: RL Training

After completing the previous steps, you can train RAPO for computational and knowledge-intensive reasoning tasks using Qwen2.5-7B-Instruct:

```bash
conda activate rapo
bash ./RAPO/scripts/RAPO_7B_Reasoning.sh
```

## Acknowledgement

Codes and model implementations are referred to [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [ARPO](https://github.com/RUC-NLPIR/ARPO) project. Thanks for their great contributions!





