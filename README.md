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
