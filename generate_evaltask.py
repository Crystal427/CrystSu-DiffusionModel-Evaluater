import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image
import re
import tempfile
from datetime import datetime
import argparse
import zipfile


# 设置features_threshold
features_threshold = 0.27
ROOT = Path(__file__).parent
PATTERN_ESCAPED_BRACKET = r"\\([\(\)\[\]\{\}])"  # match `\(` and `\)`

def search_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None

OVERLAP_TABLE_PATH = search_file('overlap_tags.json', ROOT)
OVERLAP_TABLE = None

def init_overlap_table(table_path=OVERLAP_TABLE_PATH):
    global OVERLAP_TABLE
    if OVERLAP_TABLE is not None:
        return True
    try:
        with open(table_path, 'r') as f:
            table = json.load(f)
        table = {entry['query']: (set(entry.get("has_overlap") or []), set(entry.get("overlap_tags") or [])) for entry in table}
        table = {k: v for k, v in table.items() if len(v[0]) > 0 or len(v[1]) > 0}
        OVERLAP_TABLE = table
        return True
    except Exception as e:
        OVERLAP_TABLE = None
        print(f'failed to read overlap table: {e}')
        return False

def unescape(s):
    return re.sub(PATTERN_ESCAPED_BRACKET, r'\1', s)

def fmt2danbooru(tag):
    tag = tag.lower().replace(' ', '_').strip('_').replace(':_', ':')
    tag = unescape(tag)
    return tag

def deoverlap_tags(tag_str):
    tags = tag_str.split(", ")
    init_overlap_table()
    dan2tag = {fmt2danbooru(tag): tag for tag in tags}
    tag2dan = {v: k for k, v in dan2tag.items()}
    ovlp_table = OVERLAP_TABLE
    tags_to_remove = set()
    tagset = set(tags)
    
    for tag in tagset:
        dantag = tag2dan[tag]
        if dantag in ovlp_table and tag not in tags_to_remove:
            parents, children = ovlp_table[dantag]
            parents = {dan2tag[parent] for parent in parents if parent in dan2tag}
            children = {dan2tag[child] for child in children if child in dan2tag}
            tags_to_remove |= tagset & children
    
    deoverlaped_tags = [tag for tag in tags if tag not in tags_to_remove]
    return ", ".join(deoverlaped_tags)

def process_image(artist_folder, year_folder, filename, output_folder):
    name, ext = os.path.splitext(filename)
    
    txt_path = os.path.join(artist_folder, year_folder, name + '.txt')
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            finaltag_dan = f.read().strip()
    else:
        finaltag_dan = None

    danbooru_json = None
    json_folder = os.path.join(artist_folder, 'json')
    for ext in ['.json', '.png.json', '.jpg.json', '.jpeg.json', '.webp.json']:
        json_path = os.path.join(json_folder, name + ext)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                danbooru_json = json.load(f)
            break
    
    results_json = None
    results_path = os.path.join(artist_folder, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results_json = json.load(f)
    
    final_artist_tag = os.path.basename(artist_folder).replace('_', ' ') + ', '
    
    final_copyright_tag = ''
    if danbooru_json and 'tags_copyright' in danbooru_json:
        final_copyright_tag = ','.join(danbooru_json['tags_copyright']).replace('_', ' ') + ', '
    
    final_character_tag = ''
    if danbooru_json and 'tags_character' in danbooru_json and danbooru_json['tags_character']:
        final_character_tag = ', '.join(danbooru_json['tags_character']).replace('_', ' ') + ', '
    elif results_json and filename in results_json and 'character' in results_json[filename] and results_json[filename]['character']:
        final_character_tag = ', '.join(results_json[filename]['character'].keys()).replace('_', ' ') + ', '
    
    features_tag = set()
    
    if danbooru_json and 'tags_general' in danbooru_json:
        features_tag.update(tag.replace('_', ' ') for tag in danbooru_json['tags_general'])
    
    if results_json and filename in results_json and 'features' in results_json[filename]:
        features_tag.update(k.replace('_', ' ') for k, v in results_json[filename]['features'].items() if v > features_threshold)
    
    final_features_tag = ', '.join(sorted(features_tag))

    final_Comment = ''
    if results_json and filename in results_json and 'Comment' in results_json[filename]:
        comment = results_json[filename]['Comment']
        comment = re.sub(r'The text on the image reads.*?\.', '', comment)
        comment = re.sub(r'The image shows\s+', '', comment)
        comment = re.sub(r'The image is.*?of\s+', '', comment)
        final_Comment = comment.strip()

    final_rating_tag = ''
    if results_json and filename in results_json:
        if results_json[filename].get('is_AI'):
            final_rating_tag += 'ai-generated, '
        
        scores_by_class = results_json[filename].get('scores_by_class', {})
        if scores_by_class:
            max_class = max(scores_by_class, key=scores_by_class.get)
            if max_class != 'masterpiece':
                final_rating_tag += f'{max_class} quality, '
        
        rating = results_json[filename].get('rating', {})
        if rating:
            max_rating = max(rating, key=rating.get)
            if max_rating == 'general':
                max_rating = 'safe'
            elif max_rating == 'questionable':
                max_rating = 'nsfw'
            final_rating_tag += max_rating + ', '
        
        img_path = os.path.join(artist_folder, year_folder, filename)
        with Image.open(img_path) as img:
            width, height = img.size
            if width * height <= 589824:
                final_rating_tag += 'lowres, '
            elif width * height >= 1638400:
                final_rating_tag += 'absurdres, '
    
    additional_tags = ''
    if results_json and filename in results_json and 'additional_tags' in results_json[filename]:
        additional_tags = results_json[filename]['additional_tags'].replace('_', ' ')
    
    year_tag = ''
    if year_folder in ['new', '2022s', '2020s', '2017s', '2010s']:
        year_mapping = {
            'new': 'newest, ',
            '2022s': 'recent, ',
            '2020s': 'mid, ',
            '2017s': 'early, ',
            '2010s': 'old, '
        }
        year_tag = year_mapping.get(year_folder, '')
    
    if finaltag_dan is None:
        prefix_tags = filter(bool, [
            final_artist_tag.strip(', '),
            final_character_tag.strip(', '),
            final_copyright_tag.strip(', ')
        ])
        prefix = ", ".join(prefix_tags)

        suffix_tags = filter(bool, [
            final_features_tag,
            final_rating_tag.strip(', '),
            year_tag.strip(', '),
            additional_tags.strip(', ')
        ])
        suffix = ", ".join(suffix_tags)

        finaltag_dan = f"{prefix}, |||{suffix}"

    tags_native = [
        final_artist_tag.strip(', '),
        final_character_tag.strip(', '),
        final_copyright_tag.strip(', '),
        final_rating_tag.strip(', '),
        year_tag.strip(', '),
        additional_tags.strip(', '),
        final_Comment.strip(', ')
    ]
    finaltag_native = ", ".join(filter(bool, tags_native))

    return {
        "finaltag_dan": finaltag_dan,
        "finaltag_native": finaltag_native,
        "original_data": results_json.get(filename, {}) if results_json else {}
    }


def process_artist(artist_folder, output_folder, percentile):
    results_path = os.path.join(artist_folder, 'results.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 按文件夹分类图片
    images_by_folder = {
        'old': [],  # 2010s and 2017s
        'new': []   # 2020s, 2022s, new, unknown, undefined
    }
    
    for filename, data in results.items():
        folder = next((f for f in ['2010s', '2017s', '2020s', '2022s', 'new', 'unknown', 'undefined'] 
                       if os.path.exists(os.path.join(artist_folder, f, filename))), None)
        if folder:
            if folder in ['2010s', '2017s']:
                images_by_folder['old'].append((filename, data))
            else:
                images_by_folder['new'].append((filename, data))
    
    # 根据percentile排序每个类别的图片
    for category in images_by_folder:
        images_by_folder[category].sort(key=lambda x: x[1]['aesthetic_score'], reverse=True)
        images_by_folder[category] = images_by_folder[category][:int(len(images_by_folder[category]) * percentile)]
    
    # 选择图片
    selected_images = []
    if len(images_by_folder['old']) >= 30:
        selected_images.extend(random.sample(images_by_folder['old'], 4))
        selected_images.extend(random.sample(images_by_folder['new'], 6))
    else:
        selected_images.extend(random.sample(images_by_folder['new'], 10))
    
    # 处理选中的图片
    artist_output_folder = os.path.join(output_folder, os.path.basename(artist_folder))
    os.makedirs(os.path.join(artist_output_folder, 'OriginalPic'), exist_ok=True)
    os.makedirs(os.path.join(artist_output_folder, 'ModelGenPicDan'), exist_ok=True)
    os.makedirs(os.path.join(artist_output_folder, 'ModelGenPicComt'), exist_ok=True)
    
    tracer_data = {}
    
    for i, (filename, _) in enumerate(selected_images, start=1):
        for root, _, files in os.walk(artist_folder):
            if filename in files:
                original_path = os.path.join(root, filename)
                year_folder = os.path.basename(root)
                
                new_filename = f"{i:03d}{os.path.splitext(filename)[1]}"
                shutil.copy2(original_path, os.path.join(artist_output_folder, 'OriginalPic', new_filename))
                
                processed_data = process_image(artist_folder, year_folder, filename, output_folder)
                
                tracer_data[new_filename] = {
                    "PicDanboorutagsRatingPerEpoch": {},
                    "PicNatrualLanguagetagsRatingPerEpoch": {},
                    "Danboorutags": processed_data["finaltag_dan"],
                    "Florence2tags": processed_data["finaltag_native"],
                    "PicOriginalJsonData": processed_data["original_data"],
                    "OriginalName": filename,
                    "OriginalFolder": year_folder  # 添加这一行来记录原始文件夹
                }
                break
    
    with open(os.path.join(artist_output_folder, 'tracer.json'), 'w', encoding='utf-8') as f:
        json.dump(tracer_data, f, ensure_ascii=False, indent=2)


def main(dataset_folder, percentile, output_folder):
    with tempfile.TemporaryDirectory() as temp_dir:
        artist_folders = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
        total_artists = len(artist_folders)
        
        for artist_folder in artist_folders:
            artist_path = os.path.join(dataset_folder, artist_folder)
            process_artist(artist_path, temp_dir, percentile)
        
        # 创建meta.json
        meta_data = {
            "creation_time": datetime.now().isoformat(),
            "datasetver": "4.0",
            "total_artists": total_artists
        }
        with open(os.path.join(temp_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        # 创建zip文件
        os.makedirs(output_folder, exist_ok=True)
        zip_filename = os.path.join(output_folder, f"DatasetEval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        
        print(f"Dataset evaluation completed. Output file: {zip_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and create compressed output.")
    parser.add_argument("--dataset_folder", type=str, default="Dataset", help="Path to the dataset folder")
    parser.add_argument("--percentile", type=float, default=0.75, help="Percentile for image selection (default: 0.75)")
    parser.add_argument("--output_folder", type=str, default=".", help="Folder to save the output zip file (default: current directory)")
    
    args = parser.parse_args()
    
    main(args.dataset_folder, args.percentile, args.output_folder)