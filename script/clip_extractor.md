# Clip Extractor

## Description
This script processes `.bk2` replay files from the Mario dataset to extract gameplay clips and associated data, such as game states, RAM dumps, and visual representations.

## Requirements
- Python 3.x
- `retro`
- `pandas`
- `numpy`
- `skvideo`
- `Pillow`
- `joblib`
- `tqdm`

Install dependencies with:

```bash
pip install retro pandas numpy skvideo pillow joblib tqdm
```

## Usage

```bash
python clip_extractor.py --datapath [data_folder] --output [output_folder]
```

### Arguments
- `--datapath` (default: `.`): Root folder containing the Mario dataset.
- `--output` (default: `None`): Path to save the extracted clips.
- `--filetypes` (optional): Types of output to generate (e.g., `gif`, `mp4`, `json`).
- `--n_jobs` (default: `-1`): Number of CPU cores to use for parallel processing.
- `--verbose` (optional): Increase verbosity for logging.

Example:

```bash
python clip_extractor.py --datapath ./mario_data --output ./derivatives --filetypes gif mp4 json
```

## Output
The extracted clips and data are saved in the specified output folder, following the BIDS format convention:

```
sub-01/ses-001/beh/clips/sub-01_ses-001_run-01_level-1_scene-2_clip-0000001_beh.mp4
sub-01/ses-001/beh/savestates/sub-01_ses-001_run-01_level-1_scene-2_beh.state
```

## Notes
- Ensure `.bk2` replay files are available in the dataset.
- Scenes information should be provided in a CSV file if required.
- Default settings process all subjects and sessions unless specified.

## License
This project is licensed under the MIT License.

