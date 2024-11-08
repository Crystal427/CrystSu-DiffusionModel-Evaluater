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
import shutil
from tqdm import tqdm
import time
import requests

class SDWebUIGenerator:
    def __init__(self, host, port, model, use_https=True, max_retries=3, retry_delay=5):
        self.host = host
        self.port = port
        self.model = model
        self.use_https = use_https
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
        self.set_model(model)

    def _connect_with_retry(self):
        while True:
            try:
                api = webuiapi.WebUIApi(host=self.host, port=self.port, use_https=self.use_https)
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
    with open(tracer_path, 'r', encoding='utf-8') as f:
        tracer_data = json.load(f)
    
    image_data = tracer_data.get(image_name, {})
    return image_data.get('Danboorutags', ''), image_data.get('Florence2tags', '')

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

def update_tracer_json(artist_path, image_name, model):
    tracer_path = os.path.join(artist_path, 'tracer.json')
    with open(tracer_path, 'r', encoding='utf-8') as f:
        tracer_data = json.load(f)
    
    image_data = tracer_data.get(image_name, {})
    
    if 'PicDanboorutagsRatingPerEpoch' not in image_data:
        image_data['PicDanboorutagsRatingPerEpoch'] = {}
    if 'PicNatrualLanguagetagsRatingPerEpoch' not in image_data:
        image_data['PicNatrualLanguagetagsRatingPerEpoch'] = {}
    
    image_data['PicDanboorutagsRatingPerEpoch'][f"{image_name}_{model}"] = None
    image_data['PicNatrualLanguagetagsRatingPerEpoch'][f"{image_name}_{model}"] = None
    
    tracer_data[image_name] = image_data
    
    with open(tracer_path, 'w') as f:
        json.dump(tracer_data, f, indent=2)

def main(zip_path, host, port, model, eval_pic_num, use_https, model_marked_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, 'temp')
    meta_path = os.path.join(temp_dir, 'meta.json')

    if not os.path.exists(temp_dir) or not os.listdir(temp_dir):
        print("解压评估文件...")
        os.makedirs(temp_dir, exist_ok=True)
        unzip_file(zip_path, temp_dir)
    else:
        print("使用已存在的临时文件。")

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    
    if 'completed_models' in meta_data and model in meta_data['completed_models']:
        print(f"模型 {model} 已完成评估。")
        return

    generator = SDWebUIGenerator(host, port, model, use_https=use_https)

    current_artist = meta_data.get('current_artist', '')
    artist_folders = [folder for folder in os.listdir(temp_dir) 
                     if os.path.isdir(os.path.join(temp_dir, folder))]

    with tqdm(total=len(artist_folders), desc=f"处理模型 {model}") as pbar:
        for artist_folder in artist_folders:
            artist_path = os.path.join(temp_dir, artist_folder)
            
            if current_artist and artist_folder <= current_artist:
                pbar.update(1)
                continue

            print(f"\n处理艺术家: {artist_folder}")
            selected_images = select_images(artist_path, eval_pic_num)
            is_first_image = True

            for image_path in selected_images:
                image_name = os.path.basename(image_path)
                image_name_without_ext = os.path.splitext(image_name)[0]
                
                danbooru_tags, florence_tags = process_image(artist_path, image_name)
                
                danbooru_output_dir = os.path.join(artist_path, 'ModelGenPicDan')
                florence_output_dir = os.path.join(artist_path, 'ModelGenPicComt')
                os.makedirs(danbooru_output_dir, exist_ok=True)
                os.makedirs(florence_output_dir, exist_ok=True)
                
                danbooru_image = generator.generate(danbooru_tags)
                danbooru_output_path = os.path.join(danbooru_output_dir, 
                    f"{image_name_without_ext}_{model_marked_name}.png")
                danbooru_image.save(danbooru_output_path, format='PNG')
                
                if is_first_image:
                    florence_image = generator.generate(florence_tags)
                    florence_output_path = os.path.join(florence_output_dir,
                        f"{image_name_without_ext}_{model_marked_name}.png")
                    florence_image.save(florence_output_path, format='PNG')
                    is_first_image = False
                
                update_tracer_json(artist_path, image_name, model)
            
            update_meta_json(meta_path, model, artist_folder)
            pbar.update(1)
    
    mark_model_as_completed(meta_path, model)
    
    output_zip_path = os.path.join(script_dir, zip_path)
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)

    shutil.rmtree(temp_dir)
    print("评估完成，临时文件已清理")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估 SD 模型")
    parser.add_argument("--zip_path", help="包含数据集的 ZIP 文件路径")
    parser.add_argument("--host", default="localhost", help="SD Web UI API 的主机")
    parser.add_argument("--port", type=int, default=7860, help="SD Web UI API 的端口")
    parser.add_argument("--model", help="用于生成的模型名称")
    parser.add_argument("--eval_pic_num", default=8, type=int, help="要评估的图片数量")
    parser.add_argument("--use_https", action='store_true', help="使用 HTTPS 连接到 SD Web UI API")
    parser.add_argument("--model_marked_name", help="用于保存图片文件名的模型标记名称")
    
    args = parser.parse_args()
    
    model_marked_name = args.model_marked_name if args.model_marked_name else args.model
    
    main(args.zip_path, args.host, args.port, args.model, min(args.eval_pic_num, 10), args.use_https, model_marked_name)