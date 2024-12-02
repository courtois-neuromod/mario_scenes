import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
import umap.umap_ as umap
import plotly.express as px
import plotly.io as pio

# Set the default renderer to open in your browser
pio.renderers.default = 'browser'

# Directory containing the GIF files
gif_directory = '/home/hyruuk/GitHub/neuromod/mario/pattern_clips'  # Replace with your folder path

# Function to extract features from a GIF
def extract_features(gif_path, num_frames=10):
    try:
        with Image.open(gif_path) as img:
            frames = []
            for i in range(num_frames):
                try:
                    img.seek(i)
                    frame = img.convert('RGB').resize((64, 64))
                    frames.append(np.array(frame).flatten())
                except EOFError:
                    break
            if frames:
                # Average over frames
                return np.mean(frames, axis=0)
            else:
                return None
    except Exception as e:
        print(f"Error processing {gif_path}: {e}")
        return None

# Load and process all GIF files
gif_files = [os.path.join(gif_directory, f) for f in os.listdir(gif_directory) if f.lower().endswith('.gif')]

features = []
file_paths = []
for idx, gif_path in enumerate(gif_files):
    print(f"Processing file {idx+1}/{len(gif_files)}: {gif_path}")
    feature = extract_features(gif_path)
    if feature is not None:
        features.append(feature)
        file_paths.append(gif_path)

# Convert features to a NumPy array
features = np.array(features)

# Optional: Reduce dimensionality before UMAP using PCA to speed up UMAP
pca = PCA(n_components=50)
features_pca = pca.fit_transform(features)

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
embedding = reducer.fit_transform(features_pca)

# Create a DataFrame for visualization
df = pd.DataFrame({
    'x': embedding[:, 0],
    'y': embedding[:, 1],
    'gif_path': file_paths
})

# Function to encode GIFs to base64 for embedding in hover tooltips
import base64
def encode_gif(gif_path):
    with open(gif_path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode('utf-8')
    return f'data:image/gif;base64,{encoded}'

# Add encoded GIFs to the DataFrame
df['gif_data'] = df['gif_path'].apply(encode_gif)

# Create the interactive plot using Plotly
fig = px.scatter(
    df, x='x', y='y',
    hover_data={'gif_path': False, 'gif_data': True},
    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
    title='UMAP Clustering of Super Mario Bros GIFs'
)

# Customize hover template to display the GIF
fig.update_traces(
    hovertemplate='<b>GIF Path:</b> %{customdata[0]}<br>',
    customdata=df[['gif_path']].values
)

# Show the plot
fig.show()
