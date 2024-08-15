# CrystSu_DiffusionModel_Evaluater
Some tools to help us evaluate our Stable diffusion models


# How to use?  
## Generate Eval Task 
```
python generate_evaltask.py --dataset_folder Path/to/Your/Dataset
```
You will get dataset.eval on script folder,Dataset Folder format is defined by our own dataset.  
script will randomly select 10 image and their tags.  

## Run Eval Task  
```  
python evaluater.py --eval_path Path/to/Your/GeneratedEvalTaskFile  
```
You can use --host to specific model runner. gradio interface is also supported.  
For a large dataset, evaluation may take a lot of time. Muilt-threading support is currently developing.  

## Visualization Eval-result  
```
python gradio_interface.py  --eval_path Path/to/Your/GeneratedEvalTaskFile
```
