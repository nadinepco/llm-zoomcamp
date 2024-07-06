# 2. Open-Source LLMs
## Table of Contents

- [Using a GPU in Saturn Cloud](#saturn-cloud)
- [Google Flan T5](#23-google-flan-t5)
- [Phi-3-mini](#24-phi-3-mini)
- [Mistral-7B](#25-mistral-7b)
- [Running LLMs on CPU](#26-ollama---running-llms-on-cpu)



## 2.2 Saturn Cloud

Most open-source LLMs require a GPU, hence we will use Saturn Cloud. But an alternative is Google Collab.

### 2.2.1 Platform
1. Secrets <br/>
    -Place where we can add tokens
2. Git access </br>
    -Add Git SSH Key to be able to access the models you create in Saturn Cloud to Git
    -User -> Git SSH Key -> Add key to your Github settings
### 2.2.2 Creating a notebook
- Resources -> New Python Server
- GPU -> saturn-python-llm
- install packages found in the lecture
- Add llm-zoomcamp-saturncloud Git repo
- Add HF_Token to secrets

> [!NOTE]
> Hugging face uses home as default directory. Check if you have enough space on it by typing
```!df -h``` 

Change HF home directory 
```
import os
os.environ['HF_HOME'] = '/run/cache/'
```

## 2.3 Google FLAN T5
[Google Flan T5 xl info](https://huggingface.co/google/flan-t5-xl)

- Run the sample code for GPU in Jupyter NB

## 2.4 Phi-3-mini
[Microsoft Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)

## 2.5 Mistral-7B
[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
https://huggingface.co/docs/transformers/en/llm_tutorial

- You will need to accept the agreement in the huggingface mistral website to get access to this model. You will also need to provide the HF_TOKEN
```
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])
```

## 2.6 Ollama - Running LLMs on CPU
### Install [Ollama](https://github.com/ollama/ollama)
For Linux:
```
curl -fsSL https://ollama.com/install.sh | sh

ollama start
ollama pull phi3
ollama run phi3
```

### Run Ollama with Docker
```
docker run -it \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama
```


