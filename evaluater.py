import os
import json
import zipfile
import webuiapi
from PIL import Image
import pandas as pd
import numpy as np
from functools import lru_cache
from huggingface_hub import hf_hub_download
from huggingface_hub import HfFileSystem
from natsort import natsorted
from imgutils.generic import classify_predict_score
import shutil
from tqdm import tqdm
import time
import requests

class SDWebUIGenerator:
    def __init__(self, host, port, model, use_https=True, max_retries=3, retry_delay=5):
        self.host = host
        self.port = port
        self.model = model
        self.use_https = use_https  # 添加 use_https 作为实例属性
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api = self._connect_with_retry()
        
        self.negative_prompt = "lowres,bad hands,worst quality,watermark,censored,jpeg artifacts"
        self.cfg_scale = 4.5
        self.steps = 37
        self.sampler_name = 'DPM++ 2M'
        self.scheduler = 'SGM Uniform'
        self.width = 1024
        self.height = 1024
        self.seed = 47
        self.api = self._connect_with_retry()
        self.set_model(model)

    def _connect_with_retry(self):
        while True:
            try:
                api = webuiapi.WebUIApi(host=self.host, port=self.port, use_https=self.use_https)  # 使用 self.use_https
                # Test the connection
                api.util_get_model_names()
                print("Successfully connected to the SD Web UI API")
                return api
            except Exception as e:
                print(f"Connection attempt failed: {str(e)}")
                print("Retrying in 1 second...")
                time.sleep(1)



    def set_model(self, model):
        self.api.util_set_model(model)
        print("Model set to:" + model)

    def generate(self, prompt):
        result = self.api.txt2img(
            prompt=prompt,
            steps=self.steps,
            negative_prompt=self.negative_prompt,
            cfg_scale=self.cfg_scale,
            sampler_name=self.sampler_name,
            scheduler=self.scheduler,
            width=self.width,
            height=self.height,
            seed=self.seed
        )
        return result.image

def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def select_images(artist_path, eval_pic_num):
    original_pic_path = os.path.join(artist_path, 'OriginalPic')
    image_files = [f for f in os.listdir(original_pic_path) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    image_files.sort()
    return [os.path.join(original_pic_path, f) for f in image_files[:eval_pic_num]]

def process_image(artist_path, image_name):
    tracer_path = os.path.join(artist_path, 'tracer.json')
    with open(tracer_path, 'r',encoding='utf-8') as f:
        tracer_data = json.load(f)
    
    image_data = tracer_data.get(image_name, {})
    return image_data.get('Danboorutags', ''), image_data.get('Florence2tags', '')

hf_fs = HfFileSystem()
_REPOSITORY = 'deepghs/anime_aesthetic'
_DEFAULT_MODEL = 'swinv2pv3_v0_448_ls0.2_x'
_MODELS = natsorted([
    os.path.dirname(os.path.relpath(file, _REPOSITORY))
    for file in hf_fs.glob(f'{_REPOSITORY}/*/model.onnx')
])

LABELS = ["worst", "low", "normal", "good", "great", "best", "masterpiece"]

@lru_cache()
def _get_mark_table(model):
    df = pd.read_csv(hf_hub_download(
        repo_id=_REPOSITORY,
        repo_type='model',
        filename=f'{model}/samples.csv',
    ))
    df = df.sort_values(['score'])
    df['cnt'] = list(range(len(df)))
    df['final_score'] = df['cnt'] / len(df)
    x = np.concatenate([[0.0], df['score'], [6.0]])
    y = np.concatenate([[0.0], df['final_score'], [1.0]])
    return x, y

def _get_percentile(x, y, v):
    idx = np.searchsorted(x, np.clip(v, a_min=0.0, a_max=6.0))
    if idx < x.shape[0] - 1:
        x0, y0 = x[idx], y[idx]
        x1, y1 = x[idx + 1], y[idx + 1]
        return np.clip((v - x0) / (x1 - x0) * (y1 - y0) + y0, a_min=0.0, a_max=1.0)
    else:
        return y[idx]

def _fn_predict(image, model):
    scores = classify_predict_score(
        image=image,
        repo_id=_REPOSITORY,
        model_name=model,
    )
    weighted_mean = sum(i * scores[label] for i, label in enumerate(LABELS))
    x, y = _get_mark_table(model)
    percentile = _get_percentile(x, y, weighted_mean)
    return weighted_mean, percentile, scores

def process_image_file(image, aesthetic_model=_DEFAULT_MODEL):
    base_size = 1024
    img_ratio = image.size[0] / image.size[1]  # width / height
    if img_ratio > 1:  # Width is greater than height
        new_size = (base_size, int(base_size / img_ratio))
    else:  # Height is greater than width or equal
        new_size = (int(base_size * img_ratio), base_size)
    image = image.resize(new_size, Image.Resampling.LANCZOS) 
    weighted_mean, percentile, scores_by_class =  _fn_predict(image, aesthetic_model)
    return image, {
            "aesthetic_score": weighted_mean,
            "percentile": percentile,
            "scores_by_class": scores_by_class
        }

def update_meta_json(meta_path, current_model, current_artist):
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    
    meta_data['current_model'] = current_model
    meta_data['current_artist'] = current_artist
    
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)

def mark_model_as_completed(meta_path, model):
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    if 'completed_models' not in meta_data:
        meta_data['completed_models'] = []
    
    if model not in meta_data['completed_models']:
        meta_data['completed_models'].append(model)
    
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)

def update_tracer_json(artist_path, image_name, model, danbooru_score, florence_score):
    tracer_path = os.path.join(artist_path, 'tracer.json')
    with open(tracer_path, 'r',encoding='utf-8') as f:
        tracer_data = json.load(f)
    
    image_data = tracer_data.get(image_name, {})
    
    if 'PicDanboorutagsRatingPerEpoch' not in image_data:
        image_data['PicDanboorutagsRatingPerEpoch'] = {}
    if 'PicNatrualLanguagetagsRatingPerEpoch' not in image_data:
        image_data['PicNatrualLanguagetagsRatingPerEpoch'] = {}
    
    image_data['PicDanboorutagsRatingPerEpoch'][f"{image_name}_{model}"] = danbooru_score
    image_data['PicNatrualLanguagetagsRatingPerEpoch'][f"{image_name}_{model}"] = florence_score
    
    tracer_data[image_name] = image_data
    
    with open(tracer_path, 'w') as f:
        json.dump(tracer_data, f, indent=2)



def main(zip_path, host, port, model, eval_pic_num, use_https):
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, 'temp')
    dataset_path = os.path.join(temp_dir)
    meta_path = os.path.join(temp_dir, 'meta.json')

    # Check if temp directory is empty
    if not os.path.exists(temp_dir) or not os.listdir(temp_dir):
        print("Temp directory is empty. Extracting ZIP file...")
        os.makedirs(temp_dir, exist_ok=True)
        unzip_file(zip_path, temp_dir)
    else:
        print("Temp directory is not empty. Using existing files.")

    # Check if the model has already been completed
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    if 'completed_models' in meta_data and model in meta_data['completed_models']:
        print(f"Model {model} has already been evaluated. Exiting.")
        return

    # Initialize the generator
    max_retries = 3
    for attempt in range(max_retries):
        try:
            generator = SDWebUIGenerator(host, port, model, use_https=use_https)  # 传递 use_https 参数
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to connect to the SD Web UI API: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 1 second...")
                time.sleep(1)
            else:
                print(f"Failed to connect after {max_retries} attempts. Exiting.")
                return

    # Get the current progress from meta.json
    current_artist = meta_data.get('current_artist', '')

    # Get the list of all artist folders
    artist_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    # Create a tqdm progress bar
    pbar = tqdm(total=len(artist_folders), desc=f"Processing artists for model {model}", position=0, leave=True)

    for artist_folder in artist_folders:
        artist_path = os.path.join(dataset_path, artist_folder)
        
        # Skip artists that have been processed
        if current_artist and artist_folder <= current_artist:
            pbar.update(1)
            continue

        print(f"\nProcessing artist: {artist_folder}")
        selected_images = select_images(artist_path, eval_pic_num)
        
        # 标记是否为当前艺术家的第一张图片
        is_first_image = True

        for image_path in selected_images:
            image_name = os.path.basename(image_path)
            image_name_without_ext, _ = os.path.splitext(image_name)
            
            print(f"  Processing image: {image_name}")
            danbooru_tags, florence_tags = process_image(artist_path, image_name)
            
            # Create output directories if they don't exist
            danbooru_output_dir = os.path.join(artist_path, 'ModelGenPicDan')
            florence_output_dir = os.path.join(artist_path, 'ModelGenPicComt')
            os.makedirs(danbooru_output_dir, exist_ok=True)
            os.makedirs(florence_output_dir, exist_ok=True)
            
            # Generate and save Danbooru image
            for attempt in range(max_retries):
                try:
                    danbooru_image = generator.generate(danbooru_tags)
                    danbooru_output_path = os.path.join(danbooru_output_dir, f"{image_name_without_ext}_{model}.png")
                    danbooru_image.save(danbooru_output_path, format='PNG')
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed to generate Danbooru image: {str(e)}")
                    if attempt < max_retries - 1:
                        print("Retrying in 1 second...")
                        time.sleep(1)
                    else:
                        print(f"Failed to generate Danbooru image after {max_retries} attempts. Skipping.")
            
            # 仅为第一张图片生成 Florence 图像
            if is_first_image:
                for attempt in range(max_retries):
                    try:
                        florence_image = generator.generate(florence_tags)
                        florence_output_path = os.path.join(florence_output_dir, f"{image_name_without_ext}_{model}.png")
                        florence_image.save(florence_output_path, format='PNG')
                        break
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed to generate Florence image: {str(e)}")
                        if attempt < max_retries - 1:
                            print("Retrying in 1 second...")
                            time.sleep(1)
                        else:
                            print(f"Failed to generate Florence image after {max_retries} attempts. Skipping.")
                # 生成完第一张图片后，将标记设为 False
                is_first_image = False
            else:
                florence_score = {"aesthetic_score": None}  # 或根据需要设置默认值
                print(f"  Skipping Florence image generation for {image_name}")
            
            # 评估图片
            _, danbooru_score = process_image_file(Image.open(danbooru_output_path))
            if is_first_image:
                _, florence_score = process_image_file(Image.open(florence_output_path))
            else:
                florence_score = {"aesthetic_score": None}  # 或根据需要设置默认值
            
            # 更新 tracer.json
            update_tracer_json(artist_path, image_name, model, 
                               danbooru_score['aesthetic_score'], 
                               florence_score['aesthetic_score'] if florence_score else None)
            
            print(f"    Completed processing {image_name}")
        
        # Update meta.json after processing each artist
        update_meta_json(meta_path, model, artist_folder)
        
        print(f"Completed processing {artist_folder}")
        pbar.update(1)
    
    pbar.close()
    
    # Mark the model as completed
    mark_model_as_completed(meta_path, model)
    print(f"Completed evaluation for model {model}")

    # Repack the temp folder contents into a new ZIP file
    output_zip_path = os.path.join(script_dir, zip_path)
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)

    print(f"Repacked temp folder contents into {output_zip_path}")

    # Optionally, remove the temp folder after zipping
    shutil.rmtree(temp_dir)
    print("Removed temp folder")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SD models")
    parser.add_argument("--zip_path", help="Path to the ZIP file containing the dataset")
    parser.add_argument("--host", default="localhost", help="Host for the SD Web UI API")
    parser.add_argument("--port", type=int, default=7860, help="Port for the SD Web UI API")
    parser.add_argument("--model", help="Model name to use for generation")
    parser.add_argument("--eval_pic_num", default=4, type=int, help="Number of pictures to evaluate (max 10)")
    parser.add_argument("--use_https", action='store_true', help="Use HTTPS for connecting to SD Web UI API")  # 添加 use_https 参数
    
    args = parser.parse_args()
    
    main(args.zip_path, args.host, args.port, args.model, min(args.eval_pic_num, 10), args.use_https) 