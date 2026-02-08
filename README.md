# MammoWise
A Vision Language Model-based Local Mammogram Report Generation Pipeline


## 1) Installations
- Install [Git](https://git-scm.com/downloads) 

- Install [Python 3.11.7](https://www.python.org/downloads/release/python-3117/) 

- Install [Ollama](https://ollama.com/)

- You can pull the following models beforehand from your command prompt, since they are used in the given version of the tool:

```
ollama pull alibayram/medgemma:4b
```

The other models are: 
- qwen2.5vl:7b
- rohithbojja/llava-med-v1.6:latest
- Models you want to use
  
## 2) Clone the git repository
```
git clone https://github.com/RaiyanJahangir/MammoWise.git
```

## 3) Go to the root directory of the project
```
cd MammoWise
```

## 4) Create a virtual environment 
```
python3.11 -m venv myenv 
```

## 5) Activate the virtual environment 
```
source myenv/bin/activate
```

## 6) Install all the necessary packages and libraries
```
pip install -r requirements.txt
```

## 7) Download the datasets
- Donwload Vindr-Mammo dataset from any of the two Links: [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/)[Kaggle](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png).

- Donwload DMID dataset from [Figshare](https://figshare.com/authors/Parita_Oza/17353984).


## 8) Set your filepaths and other credentials in config.py

## 9) Preprocess the datasets (You can use your own datasets)
### 9a) Preprocess Vindr-Mammo
```
python3 preprocessNmerge_vindr.py
```

### 9b) Preprocess DMID
```
python3 preprocess_dmid_.py
```

### 9c) Create multimodal embeddings for RAG
```
python3 populate_db.py
```

## 10) Prompt Engineering
Set all the filepaths and variables in config.py before running this code
```
python3 ollama_infer.py
```

## 11) Preprocessing for Fine-tuning
### 11a) Data Augmentation for Fine-tuning
```
python3 data_augmentation.py
```

### 11b) Create data dictionary for each category of Fine-tuning 
#### For checking only one type, just train that.
For Multi-task
```
python3 data_preprocess.py
```

For BIRADS only
```
python3 data_preprocess_birads.py
```

For Density only
```
python3 data_preprocess_density.py
```

For Calcification only
```
python3 data_preprocess_calcification.py
```

For Mass only
```
python3 data_preprocess_mass.py
```

For Asymmetry only
```
python3 data_preprocess_asymmetry.py
```

For Suspicion only
```
python3 data_preprocess_suspicion.py
```

## 12) Finetune MedGemma model
For Multi-task
```
python3 finetune_all.py
```

For BIRADS only
```
python3 finetune_birads.py
```

For Density only
```
python3 finetune_density.py
```

For Calcification only
```
python3 finetune_calcification.py
```

For Mass only
```
python3 finetune_mass.py
```

For Asymmetry only
```
python3 finetune_asymmetry.py
```

For Suspicion only
```
python3 finetune_suspicion.py
```

## 13) Inference from the trained MedGemma models
For Multi-task
```
python3 generate_all.py
```

For BIRADS only
```
python3 generate_birads.py
```

For Density only
```
python3 generate_density.py
```

For Calcification only
```
python3 generate_calcification.py
```

For Mass only
```
python3 generate_mass.py
```

For Asymmetry only
```
python3 generate_asymmetry.py
```

For Suspicion only
```
python3 generate_suspicion.py
```

## 14) Evaluate all the generated results
```
python3 evaluate_all.py
```
You can also run this code after prompt engineering step.


## 15) Run the whole project as a Tool
```
python3 MammoWise.py
```

## 16) Deactivate Virtual Environment and wrap up
```
deactivate
```
