# CrystSu_DiffusionModel_Evaluater
A tool that provides automatic generation of Diffusion Model inferred images, and a convenient HumanEval GUI to help us evaluate our model's progress in learning artistic styles.

Mainly use [dghs-imgutils](https://github.com/deepghs/imgutils) to evaluate images.

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

After load the dataset,You can use keyboard shortcuts to quickly mark artists. There are 5 marking categories:  

1. OverFit
2. Well-Trained
3. Need-Train
4. Unfit",
5. Bad Artist  

Mark will be stored at human_feedback.json  

**Currently Working on gradio interface**