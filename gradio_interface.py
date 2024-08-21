import gradio as gr
import os
import json
import zipfile
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

def unzip_file(file_path, unzip_folder):
    if not os.path.exists(unzip_folder):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)
    return unzip_folder

def get_artists(folder_path):
    return [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

def get_picnums(artist_folder):
    original_pic_folder = os.path.join(artist_folder, "OriginalPic")
    return [os.path.splitext(f)[0] for f in os.listdir(original_pic_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]

def load_tracer_data(artist_folder):
    with open(os.path.join(artist_folder, "tracer.json"), 'r') as f:
        return json.load(f)

def get_original_pic_extension(artist_folder, picnum):
    original_pic_folder = os.path.join(artist_folder, "OriginalPic")
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        if os.path.exists(os.path.join(original_pic_folder, f"{picnum}{ext}")):
            return ext
    return None

def get_image_path(folder, filename):
    full_path = os.path.join(folder, filename)
    if os.path.exists(full_path):
        return full_path
    logging.warning(f"Image not found: {full_path}")
    return None

def get_model_epochs(artist_folder, picnum):
    tracer_data = load_tracer_data(artist_folder)
    pic_data = next((data for key, data in tracer_data.items() if key.startswith(f"{picnum}.")), None)
    if pic_data is None:
        return []
    
    epochs = set()
    for rating_data in [pic_data["PicDanboorutagsRatingPerEpoch"], pic_data["PicNatrualLanguagetagsRatingPerEpoch"]]:
        epochs.update(epoch.split('_')[1] for epoch in rating_data.keys())
    
    return sorted(list(epochs))

def create_plot(data, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, values in data.items():
        epochs = sorted(values.keys(), key=lambda x: x.split('_')[1])
        ax.plot(range(len(epochs)), [values[epoch] for epoch in epochs], label=key)
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels([epoch.split('_')[1] for epoch in epochs], rotation=45)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rating')
    ax.legend()
    plt.tight_layout()
    return fig

def update_display(unzip_folder, artist, picnum):
    artist_folder = os.path.join(unzip_folder, artist)
    
    # Get the original picture extension
    original_ext = get_original_pic_extension(artist_folder, picnum)
    if original_ext is None:
        return [None], [], [], None, None, None, "Error: Original picture not found."
    
    # Load images
    original_pic = get_image_path(os.path.join(artist_folder, "OriginalPic"), f"{picnum}{original_ext}")
    
    epochs = get_model_epochs(artist_folder, picnum)
    comt_pics = [(get_image_path(os.path.join(artist_folder, "ModelGenPicComt"), f"{picnum}_{epoch}.png"), f"{picnum}_{epoch}.png") for epoch in epochs]
    dan_pics = [(get_image_path(os.path.join(artist_folder, "ModelGenPicDan"), f"{picnum}_{epoch}.png"), f"{picnum}_{epoch}.png") for epoch in epochs]
    
    logging.info(f"Original pic path: {original_pic}")
    logging.info(f"Comt pic paths: {comt_pics}")
    logging.info(f"Dan pic paths: {dan_pics}")
    
    # Filter out None values
    comt_pics = [pic for pic in comt_pics if pic[0] is not None]
    dan_pics = [pic for pic in dan_pics if pic[0] is not None]
    
    if not original_pic:
        return [None], [], [], None, None, None, "Error: Original picture not found."
    
    # Load tracer data
    tracer_data = load_tracer_data(artist_folder)
    pic_data = next((data for key, data in tracer_data.items() if key.startswith(f"{picnum}.")), None)
    if pic_data is None:
        return [(original_pic, os.path.basename(original_pic))], comt_pics, dan_pics, None, None, None, "Error: Picture data not found in tracer.json."
    
    # Create plots
    danbooru_plot = create_plot({picnum: pic_data["PicDanboorutagsRatingPerEpoch"]}, "Danbooru Tags Rating")
    natural_plot = create_plot({picnum: pic_data["PicNatrualLanguagetagsRatingPerEpoch"]}, "Natural Language Tags Rating")
    
    # Calculate difference plot
    diff_data = {}
    for epoch_key in pic_data["PicDanboorutagsRatingPerEpoch"]:
        epoch_name = epoch_key.split('_')[1]
        diff_data[epoch_key] = abs(pic_data["PicDanboorutagsRatingPerEpoch"][epoch_key] - pic_data["PicNatrualLanguagetagsRatingPerEpoch"][epoch_key])
    diff_plot = create_plot({f"{picnum} Difference": diff_data}, "Absolute Difference between Ratings")
    
    # Prepare text information
    text_info = f"""
    Danbooru tags: {pic_data['Danboorutags']}
    
    Florence2 tags: {pic_data['Florence2tags']}
    
    Original Name: {pic_data['OriginalName']}
    
    Original Folder: {pic_data['OriginalFolder']}
    """
    
    return [(original_pic, os.path.basename(original_pic))], comt_pics, dan_pics, danbooru_plot, natural_plot, diff_plot, text_info

def create_interface(unzip_folder):
    artists = get_artists(unzip_folder)
    
    with gr.Blocks(css="""
        .original-image img {
            object-fit: contain;
            width: 100%;
            height: 100%;
            max-height: 200px;
        }
        .gallery-item {
            position: relative;
        }
        .gallery-item .caption {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.5);
            color: white;
            padding: 5px;
            font-size: 12px;
            text-align: center;
            word-wrap: break-word;
        }
    """) as demo:
        gr.Markdown("# CrystSu-Model Visualize Analysis")
        
        with gr.Row():
            artist_dropdown = gr.Dropdown(choices=artists, label="Select Artist")
            picnum_dropdown = gr.Dropdown(label="Select Picture Number")
        
        with gr.Row():
            original_image = gr.Image(
                label="Original Image", 
                show_label=True, 
                type="filepath", 
                interactive=False,
                container=False,
                elem_classes=["original-image"]
            )
        
        with gr.Row():
            comt_images = gr.Gallery(label="ModelGenPicComt Images", show_label=True, columns=[3], rows=[1], height="auto", object_fit="contain")
        
        with gr.Row():
            dan_images = gr.Gallery(label="ModelGenPicDan Images", show_label=True, columns=[3], rows=[1], height="auto", object_fit="contain")
        
        with gr.Row():
            danbooru_plot = gr.Plot(label="Danbooru Tags Rating")
            natural_plot = gr.Plot(label="Natural Language Tags Rating")
            diff_plot = gr.Plot(label="Absolute Difference between Ratings")
        
        text_info = gr.Textbox(label="Image Information", lines=10)
        
        def update_picnums(artist):
            artist_folder = os.path.join(unzip_folder, artist)
            picnums = get_picnums(artist_folder)
            return gr.update(choices=picnums)
        
        artist_dropdown.change(update_picnums, inputs=[artist_dropdown], outputs=[picnum_dropdown])
        
        def on_select(artist, picnum):
            original, comt, dan, danbooru, natural, diff, info = update_display(unzip_folder, artist, picnum)
            return (
                original[0][0] if original else None,  # Extract the first (and only) image path
                [(img, caption) for img, caption in comt],
                [(img, caption) for img, caption in dan],
                danbooru,
                natural,
                diff,
                info
            )
        
        picnum_dropdown.change(
            on_select,
            inputs=[artist_dropdown, picnum_dropdown],
            outputs=[original_image, comt_images, dan_images, danbooru_plot, natural_plot, diff_plot, text_info]
        )
    
    return demo

if __name__ == "__main__":
    # Specify the path to your eval file and the unzip folder
    eval_file_path = "filename.eval"
    unzip_folder = r"F:\CrystSu-DiffusionModel-Evaluater\DatasetEval_20240813_202059"
    
    # Unzip the file
    unzip_file(eval_file_path, unzip_folder)
    
    # Create and launch the interface
    demo = create_interface(unzip_folder)
    demo.launch(share=True)