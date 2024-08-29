import gradio as gr
import os
import json
import zipfile
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)

def unzip_file(file_path, unzip_folder):
    if not os.path.exists(unzip_folder):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)
    return unzip_folder

def get_artists(folder_path):
    return sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])

def get_picnums(artist_folder):
    original_pic_folder = os.path.join(artist_folder, "OriginalPic")
    return sorted([os.path.splitext(f)[0] for f in os.listdir(original_pic_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))])

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

def load_tracer_data(artist_folder):
    with open(os.path.join(artist_folder, "tracer.json"), 'r') as f:
        return json.load(f)

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

def create_all_images_plot(tracer_data, plot_type):
    fig, ax = plt.subplots(figsize=(12, 6))
    for image, data in tracer_data.items():
        if plot_type in data:
            epochs = sorted(data[plot_type].keys(), key=lambda x: x.split('_')[1])
            values = [data[plot_type][epoch] for epoch in epochs]
            ax.plot(range(len(epochs)), values, label=image.split('.')[0])
    
    ax.set_title(f"All Images - {plot_type}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rating')
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([epoch.split('_')[1] for epoch in epochs], rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def create_kde_plot(data, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    for epoch, values in data.items():
        sns.kdeplot(values, shade=True, ax=ax, label=f"Epoch {epoch}")
    ax.set_title(title)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    return fig

def get_all_ratings(tracer_data, rating_type):
    all_ratings = {}
    for image_data in tracer_data.values():
        for epoch, rating in image_data.get(rating_type, {}).items():
            epoch_num = epoch.split('_')[1]
            if epoch_num not in all_ratings:
                all_ratings[epoch_num] = []
            all_ratings[epoch_num].append(rating)
    return all_ratings


def get_next_artist(artists, current_artist):
    try:
        index = artists.index(current_artist)
        return artists[index + 1] if index < len(artists) - 1 else None
    except ValueError:
        return None

def get_prev_artist(artists, current_artist):
    try:
        index = artists.index(current_artist)
        return artists[index - 1] if index > 0 else None
    except ValueError:
        return None

def on_next_click(unzip_folder, current_artist, current_picnum):
    artists = get_artists(unzip_folder)
    picnums = get_picnums(os.path.join(unzip_folder, current_artist))
    
    current_index = picnums.index(current_picnum)
    if current_index < len(picnums) - 1:
        next_picnum = picnums[current_index + 1]
        return current_artist, next_picnum, ""
    else:
        next_artist = get_next_artist(artists, current_artist)
        if next_artist:
            next_picnums = get_picnums(os.path.join(unzip_folder, next_artist))
            return next_artist, next_picnums[0], ""
        else:
            return current_artist, current_picnum, "This is the last picture of the last artist."

def on_prev_click(unzip_folder, current_artist, current_picnum):
    artists = get_artists(unzip_folder)
    picnums = get_picnums(os.path.join(unzip_folder, current_artist))
    
    current_index = picnums.index(current_picnum)
    if current_index > 0:
        prev_picnum = picnums[current_index - 1]
        return current_artist, prev_picnum, ""
    else:
        prev_artist = get_prev_artist(artists, current_artist)
        if prev_artist:
            prev_picnums = get_picnums(os.path.join(unzip_folder, prev_artist))
            return prev_artist, prev_picnums[-1], ""
        else:
            return current_artist, current_picnum, "This is the first picture of the first artist."




def update_display(unzip_folder, artist, picnum):
    artist_folder = os.path.join(unzip_folder, artist)
    
    original_ext = get_original_pic_extension(artist_folder, picnum)
    if original_ext is None:
        return [None], [], [], None, None, None, None, "Error: Original picture not found.", ""
    
    original_pic = get_image_path(os.path.join(artist_folder, "OriginalPic"), f"{picnum}{original_ext}")
    
    epochs = get_model_epochs(artist_folder, picnum)
    comt_pics = [(get_image_path(os.path.join(artist_folder, "ModelGenPicComt"), f"{picnum}_{epoch}.png"), f"{picnum}_{epoch}.png") for epoch in epochs]
    dan_pics = [(get_image_path(os.path.join(artist_folder, "ModelGenPicDan"), f"{picnum}_{epoch}.png"), f"{picnum}_{epoch}.png") for epoch in epochs]
    
    logging.info(f"Original pic path: {original_pic}")
    logging.info(f"Comt pic paths: {comt_pics}")
    logging.info(f"Dan pic paths: {dan_pics}")
    
    def sort_key(filename):
        parts = filename[1].split('_')[1].split('-')
        model_version = float(parts[0])
        epoch_number = int(parts[1].replace('e.png', ''))
        return (-model_version, -epoch_number)
    
    comt_pics = sorted([pic for pic in comt_pics if pic[0] is not None], key=sort_key)
    dan_pics = sorted([pic for pic in dan_pics if pic[0] is not None], key=sort_key)
    
    if not original_pic:
        return [None], [], [], None, None, None, None, "Error: Original picture not found.", ""
    
    tracer_data = load_tracer_data(artist_folder)
    pic_data = next((data for key, data in tracer_data.items() if key.startswith(f"{picnum}.")), None)
    if pic_data is None:
        return [(original_pic, os.path.basename(original_pic))], comt_pics, dan_pics, None, None, None, None, "Error: Picture data not found in tracer.json.", ""
    
    danbooru_plot = create_plot({picnum: pic_data["PicDanboorutagsRatingPerEpoch"]}, "Danbooru Tags Rating")
    natural_plot = create_plot({picnum: pic_data["PicNatrualLanguagetagsRatingPerEpoch"]}, "Natural Language Tags Rating")
    
    diff_data = {}
    for epoch_key in pic_data["PicDanboorutagsRatingPerEpoch"]:
        epoch_name = epoch_key.split('_')[1]
        diff_data[epoch_key] = abs(pic_data["PicDanboorutagsRatingPerEpoch"][epoch_key] - pic_data["PicNatrualLanguagetagsRatingPerEpoch"][epoch_key])
    diff_plot = create_plot({f"{picnum} Difference": diff_data}, "Absolute Difference between Ratings")
    
    all_images_plot = create_all_images_plot(tracer_data, "PicDanboorutagsRatingPerEpoch")
    
    text_info = f"""
    Danbooru tags: {pic_data['Danboorutags']}
    
    NaturalLanguage tags: {pic_data['Florence2tags']}
    
    Original Name: {pic_data['OriginalName']}
    
    Original Folder: {pic_data['OriginalFolder']}
    """
    
    return [(original_pic, os.path.basename(original_pic))], comt_pics, dan_pics, danbooru_plot, natural_plot, diff_plot, all_images_plot, text_info, ""

def create_interface(unzip_folder):
    artists = get_artists(unzip_folder)
    human_feedback_data = load_human_feedback()
    
    # Get the first artist and their first pic number
    default_artist = artists[0] if artists else None
    default_picnums = get_picnums(os.path.join(unzip_folder, default_artist)) if default_artist else []
    default_picnum = default_picnums[0] if default_picnums else None
    
    shortcut_js = """
    <script>
    function shortcuts(e) {
        var event = document.all ? window.event : e;
        switch (e.target.tagName.toLowerCase()) {
            case "input":
            case "textarea":
            case "select":
            case "button":
            break;
            default:
            if (e.code == "ArrowRight") {
                document.getElementById("next-button").click();
            } else if (e.code == "ArrowLeft") {
                document.getElementById("prev-button").click();
            } else if (e.key >= "1" && e.key <= "5") {
                document.querySelector(`input[type="radio"][value="${e.key}"]`).click();
            }
        }
    }
    document.addEventListener('keyup', shortcuts, false);
    </script>
    """
    
    with gr.Blocks(css="""
        .container { max-width: 1200px; margin: auto; padding: 20px; }
        .original-image img { object-fit: contain; width: 100%; height: 100%; max-height: 300px; }
        .gallery-item { position: relative; }
        .gallery-item .caption {
            position: absolute; bottom: 0; left: 0; right: 0;
            background: rgba(0, 0, 0, 0.5); color: white;
            padding: 5px; font-size: 12px; text-align: center; word-wrap: break-word;
        }
        .navigation-buttons { display: flex; justify-content: space-between; margin-top: 20px; }
        .plot-container { display: flex; flex-wrap: wrap; justify-content: space-between; }
        .plot-item { flex-basis: calc(50% - 10px); margin-bottom: 20px; }
        .human-feedback { display: flex; flex-direction: column; margin-top: 10px; }
        .human-feedback .gradio-radio { margin-bottom: 5px; }
    """, head=shortcut_js) as demo:
        
        with gr.Tabs():
            with gr.TabItem("Image Analysis"):
                with gr.Column(elem_classes="container"):
                    gr.Markdown("# CrystSu-Model Visualize Analysis")
                    
                    with gr.Row():
                        artist_dropdown = gr.Dropdown(choices=artists, label="Select Artist", value=default_artist, scale=2)
                        picnum_dropdown = gr.Dropdown(choices=default_picnums, value=default_picnum, label="Select Picture Number", scale=2)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            original_image = gr.Image(
                                label="Original Image", 
                                show_label=True, 
                                type="filepath", 
                                interactive=False,
                                elem_classes=["original-image"]
                            )
                            
                            human_feedback = gr.Radio(
                                choices=["1. OverFit", "2. Well-Trained", "3. Need-Train", "4. Unfit", "5. Bad Artist"],
                                label="Human Feedback",
                                value=human_feedback_data.get(default_artist, ""),
                                elem_classes=["human-feedback"]
                            )
                        
                        with gr.Column(scale=2):
                            with gr.Row():
                                dan_images = gr.Gallery(label="ModelGenPicDan Images", show_label=True, columns=[3], rows=[1], height="auto", object_fit="contain")

                            with gr.Row():
                                comt_images = gr.Gallery(label="ModelGenPicComt Images", show_label=True, columns=[3], rows=[1], height="auto", object_fit="contain")
                    
                    with gr.Row(elem_classes="navigation-buttons"):
                        prev_button = gr.Button("PREV", elem_id="prev-button")
                        next_button = gr.Button("NEXT", elem_id="next-button")
                    
                    with gr.Row(elem_classes="plot-container"):
                        with gr.Column(elem_classes="plot-item"):
                            danbooru_plot = gr.Plot(label="Danbooru Tags Rating")
                        with gr.Column(elem_classes="plot-item"):
                            natural_plot = gr.Plot(label="Natural Language Tags Rating")
                        with gr.Column(elem_classes="plot-item"):
                            diff_plot = gr.Plot(label="Absolute Difference between Ratings")
                        with gr.Column(elem_classes="plot-item"):
                            all_images_plot = gr.Plot(label="All Images Danbooru Tags Rating")
                    
                    text_info = gr.Textbox(label="Image Information", lines=10)
                    gr.Markdown("Use for private model human eval")

            with gr.TabItem("Distribution Analysis"):
                with gr.Row():
                    danbooru_kde_plot = gr.Plot(label="Danbooru Tags Rating Distribution")
                    natural_kde_plot = gr.Plot(label="Natural Language Tags Rating Distribution")
                
                aesthetic_kde_plot = gr.Plot(label="Aesthetic Score Distribution")

        def update_picnums(artist):
            artist_folder = os.path.join(unzip_folder, artist)
            picnums = get_picnums(artist_folder)
            current_feedback = human_feedback_data.get(artist, "")
            return gr.update(choices=picnums, value=picnums[0] if picnums else None), current_feedback
        
        artist_dropdown.change(update_picnums, inputs=[artist_dropdown], outputs=[picnum_dropdown, human_feedback])
        
        def on_select(artist, picnum):
            original, comt, dan, danbooru, natural, diff, all_images, info, _ = update_display(unzip_folder, artist, picnum)
            
            return (
                original[0][0] if original else None,
                [(img, caption) for img, caption in comt],
                [(img, caption) for img, caption in dan],
                danbooru,
                natural,
                diff,
                all_images,
                info
            )
        
        picnum_dropdown.change(
            on_select,
            inputs=[artist_dropdown, picnum_dropdown],
            outputs=[original_image, comt_images, dan_images, danbooru_plot, natural_plot, diff_plot, all_images_plot, text_info]
        )
        
        def on_next(artist, picnum):
            next_artist, next_picnum, _ = on_next_click(unzip_folder, artist, picnum)
            if next_artist != artist:
                current_feedback = human_feedback_data.get(next_artist, "")
                return next_artist, next_picnum, current_feedback
            return next_artist, next_picnum, gr.update()
        
        def on_prev(artist, picnum):
            prev_artist, prev_picnum, _ = on_prev_click(unzip_folder, artist, picnum)
            if prev_artist != artist:
                current_feedback = human_feedback_data.get(prev_artist, "")
                return prev_artist, prev_picnum, current_feedback
            return prev_artist, prev_picnum, gr.update()
        
        next_button.click(
            on_next,
            inputs=[artist_dropdown, picnum_dropdown],
            outputs=[artist_dropdown, picnum_dropdown, human_feedback]
        )
        
        prev_button.click(
            on_prev,
            inputs=[artist_dropdown, picnum_dropdown],
            outputs=[artist_dropdown, picnum_dropdown, human_feedback]
        )
        
        def save_feedback(artist, feedback):
            update_human_feedback(artist, feedback)
            human_feedback_data[artist] = feedback  # Update local cache
        
        human_feedback.change(
            save_feedback,
            inputs=[artist_dropdown, human_feedback]
        )
        
        def update_kde_plots(artist):
            artist_folder = os.path.join(unzip_folder, artist)
            tracer_data = load_tracer_data(artist_folder)
            
            danbooru_ratings = get_all_ratings(tracer_data, "PicDanboorutagsRatingPerEpoch")
            natural_ratings = get_all_ratings(tracer_data, "PicNatrualLanguagetagsRatingPerEpoch")
            
            danbooru_kde = create_kde_plot(danbooru_ratings, "Danbooru Tags Rating Distribution")
            natural_kde = create_kde_plot(natural_ratings, "Natural Language Tags Rating Distribution")
            
            # Corrected to access aesthetic_score from PicOriginalJsonData
            aesthetic_scores = [data.get("PicOriginalJsonData", {}).get("aesthetic_score", 0) for data in tracer_data.values()]
            aesthetic_kde = create_kde_plot({"Aesthetic Score": aesthetic_scores}, "Aesthetic Score Distribution")
            
            return danbooru_kde, natural_kde, aesthetic_kde

        artist_dropdown.change(
            update_kde_plots,
            inputs=[artist_dropdown],
            outputs=[danbooru_kde_plot, natural_kde_plot, aesthetic_kde_plot]
        )
        
        # New combined function for initial load
        def initial_load(artist, picnum):
            image_outputs = on_select(artist, picnum)
            kde_outputs = update_kde_plots(artist)
            return image_outputs + kde_outputs

        # Trigger initial display for both tabs
        demo.load(
            initial_load,
            inputs=[artist_dropdown, picnum_dropdown],
            outputs=[
                original_image, comt_images, dan_images, danbooru_plot, natural_plot, 
                diff_plot, all_images_plot, text_info,
                danbooru_kde_plot, natural_kde_plot, aesthetic_kde_plot
            ]
        )
    
    return demo

def load_human_feedback():
    try:
        with open("human_feedback.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_human_feedback(data):
    with open("human_feedback.json", "w") as f:
        json.dump(data, f, indent=2)

def update_human_feedback(artist, feedback):
    data = load_human_feedback()
    data[artist] = feedback
    save_human_feedback(data)

    
if __name__ == "__main__":
    eval_file_path = r"F:\CrystSu-DiffusionModel-Evaluater\DatasetEval_20240813_202059.eval"
    unzip_folder = r"F:\CrystSu-DiffusionModel-Evaluater\temp"
    
    unzip_file(eval_file_path, unzip_folder)
    
    demo = create_interface(unzip_folder)
    demo.launch(share=True)