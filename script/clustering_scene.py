import os
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
from sklearn.decomposition import PCA
import umap.umap_ as umap
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool
from io import BytesIO
import base64

# Directory containing the GIF files
gif_directory = '/home/hyruuk/GitHub/neuromod/mario/scene_test'  # Replace with your folder path

# Function to extract features from a GIF
def extract_features(gif_path, num_frames=10):
    try:
        with Image.open(gif_path) as img:
            frames = []
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                if i >= num_frames * 4:  # Since we're taking every 4th frame
                    break
                if i % 4 == 0:  # Keep every 4th frame
                    frame = frame.convert('RGB').resize((64, 64))
                    frames.append(np.array(frame).flatten())
            if frames:
                # Average over frames
                return np.mean(frames, axis=0)
            else:
                return None
    except Exception as e:
        print(f"Error processing {gif_path}: {e}")
        return None

# Function to adjust GIF to play faster by keeping every 4th frame and adjusting durations
def encode_gif_faster(gif_path):
    try:
        with Image.open(gif_path) as img:
            # Collect frames and keep every 4th frame
            frames = [frame.copy() for i, frame in enumerate(ImageSequence.Iterator(img)) if i % 4 == 0]
            if not frames:
                print(f"No frames to process in {gif_path}.")
                return None
            # Set durations to 10ms per frame
            durations = [10] * len(frames)
            # Save adjusted GIF to a BytesIO object
            output = BytesIO()
            frames[0].save(
                output,
                format='GIF',
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                disposal=2
            )
            # Encode to base64
            encoded = base64.b64encode(output.getvalue()).decode('utf-8')
            return f'data:image/gif;base64,{encoded}'
    except Exception as e:
        print(f"Error encoding GIF {gif_path}: {e}")
        return None

# Load and process a subset of GIF files
gif_files = [
    os.path.join(gif_directory, f)
    for f in os.listdir(gif_directory)
    if f.lower().endswith('.gif')
]  # Adjust the number as needed

features = []
file_paths = []
encoded_gifs = []
for idx, gif_path in enumerate(gif_files):
    print(f"Processing file {idx+1}/{len(gif_files)}: {gif_path}")
    feature = extract_features(gif_path)
    encoded_gif = encode_gif_faster(gif_path)
    if feature is not None and encoded_gif is not None:
        features.append(feature)
        
        # Save features as an image with the same name
        save_path = gif_path.replace('.gif', '.png')
        Image.fromarray(feature.reshape(64, 64, 3).astype(np.uint8)).save(save_path)

        file_paths.append(gif_path)
        encoded_gifs.append(encoded_gif)

# Check if any features were extracted
if not features:
    print("No features extracted. Please check your GIF files.")
    exit()

# Convert features to a NumPy array
features = np.array(features)

# Optional: Reduce dimensionality before UMAP using PCA to speed up UMAP
pca = PCA(n_components=40)
features_pca = pca.fit_transform(features)

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
embedding = reducer.fit_transform(features_pca)

# Create a DataFrame for visualization
df = pd.DataFrame({
    'x': embedding[:, 0],
    'y': embedding[:, 1],
    'gif_path': file_paths,
    'gif_data': encoded_gifs
})

# Prepare data for Bokeh
source = ColumnDataSource(data=dict(
    x=df['x'],
    y=df['y'],
    gif_path=df['gif_path'],
    gifs=df['gif_data']
))

# Create the plot
hover = HoverTool(tooltips="""
    <div>
        <div>
            <img src="@gifs" alt="GIF" style="max-width:300px;"/>
        </div>
    </div>
""")

p = figure(
    title="UMAP Clustering of Super Mario Bros GIFs",
    tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
    width=800,
    height=600
)

p.circle('x', 'y', size=10, source=source)

# Save the plot to an HTML file
output_file("umap_clustering.html", title="UMAP Clustering of Super Mario Bros GIFs")

# Show the plot
show(p)
