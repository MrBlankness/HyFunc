# HyFunc

Welcome to the official GitHub repository for HyFunc!

**📢 News: this work has been accepted at the KDD 2026 !**

**If you find our project interesting or helpful, we would appreciate it if you could give us a star! Your support is a tremendous encouragement to us!**


## Overview

While agentic AI systems rely on LLMs to translate user intent into structured function calls, this process is fraught with computational redundancy, leading to high inference latency that hinders real-time applications. 
This paper identifies and addresses three key redundancies: 
(1) the redundant processing of a large library of function descriptions for every request; 
(2) the redundant use of a large, slow model to generate an entire, often predictable, token sequence; and 
(3) the redundant generation of fixed, boilerplate parameter syntax. 
We introduce HyFunc, a novel framework that systematically eliminates these inefficiencies. 
HyFunc employs a hybrid-model cascade where a large model distills user intent into a single soft token. 
This token guides a lightweight retriever to select relevant functions and directs a smaller, prefix-tuned model to generate the final call, thus avoiding redundant context processing and full-sequence generation by the large model. 
To eliminate syntactic redundancy, our dynamic templating technique injects boilerplate parameter syntax on-the-fly within an extended vLLM engine. 
To avoid potential limitations in generalization, we evaluate HyFunc on an unseen benchmark dataset, BFCL. Experimental results demonstrate that HyFunc achieves an excellent balance between efficiency and performance. 
It achieves an inference latency of 0.828 seconds, outperforming all baseline models, and reaches a performance of 80.1%, surpassing all models with a comparable parameter scale. These results suggest that HyFunc offers a more efficient paradigm for agentic AI.

![teaser](https://github.com/MrBlankness/HyFunc/blob/main/images/teaser.png)

## Install Environment

We use conda to manage the environment.
Please refer to the following steps to install the environment:

```sh
conda create -n HyFunc python=3.10 -y
conda activate HyFunc
pip install -r requirements.txt
```

## Download Datasets

Please download the training dataset from [XLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).

Please install the evaluation dataset by referring to [BFCL](https://github.com/ShishirPatil/gorilla).

## Download Models

Please download the LML from [ToolACE](https://huggingface.co/Team-ACE).

Please download the LMS from [Qwen](https://huggingface.co/Qwen).

## Running

To run the code, simply execute the following command:

Step 1: Use LML to synthesize function embeddings and soft token embeddings.

```sh
python LML_embedding.py
```

Step 2: Train the function retriever.

```sh
python function_retriever.py
```

Step 3: Train the LMS using soft tokens.

```sh
python LMS_SFT.py
```

Step 4: Perform inference using dynamic templates.

```sh
python dynamic_templating.py
```
