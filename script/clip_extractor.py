import argparse
import os
import os.path as op
import gzip
import retro
import pandas as pd
import numpy as np
import skvideo.io
from PIL import Image
from joblib import Parallel, delayed
import json
import logging
import re
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import traceback

def replay_bk2(
    bk2_path, skip_first_step=True, game=None, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """Create an iterator that replays a bk2 file, yielding frames, keypresses, annotations, sound, actions, and state."""
    movie = retro.Movie(bk2_path)
    if game is None:
        game = movie.get_game()
    logging.debug(f"Creating emulator for game: {game}")
    emulator = retro.make(game, scenario=scenario, inttype=inttype)
    emulator.initial_state = movie.get_state()
    actions = emulator.buttons
    emulator.reset()
    if skip_first_step:
        movie.step()
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(emulator.num_buttons):
                keys.append(movie.get_key(i, p))
        frame, rew, terminate, truncate, info = emulator.step(keys)
        annotations = {"reward": rew, "done": terminate, "info": info}
        state = emulator.em.get_state()
        yield frame, keys, annotations, None, actions, state
    emulator.close()
    movie.close()


def get_variables_from_replay(
    bk2_fpath, skip_first_step=True, game=None, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """Replay the bk2 file and return game variables and frames."""
    replay = replay_bk2(
        bk2_fpath, skip_first_step=skip_first_step, game=game, scenario=scenario, inttype=inttype
    )
    all_frames = []
    all_keys = []
    all_info = []
    annotations = {}
    for frame, keys, annotations, _, actions, _ in replay:
        all_keys.append(keys)
        all_info.append(annotations["info"])
        all_frames.append(frame)
    repetition_variables = reformat_info(all_info, all_keys, bk2_fpath, actions)

    if not annotations.get('done', False):
        logging.warning(f"Done condition not satisfied for {bk2_fpath}. Consider changing skip_first_step.")

    return repetition_variables, all_frames


def reformat_info(info, keys, bk2_fpath, actions):
    """Create a structured dictionary from replay info."""
    filename = op.basename(bk2_fpath)
    entities = filename.split('_')
    entities_dict = {}
    for ent in entities:
        if '-' in ent:
            key, value = ent.split('-', 1)
            entities_dict[key] = value

    repetition_variables = {
        "filename": bk2_fpath,
        "level": entities_dict.get('level'),
        "subject": entities_dict.get('sub'),
        "session": entities_dict.get('ses'),
        "repetition": entities_dict.get('run'),
        "actions": actions,
    }

    for key in info[0].keys():
        repetition_variables[key] = []
    for button in actions:
        repetition_variables[button] = []

    for frame_idx, frame_info in enumerate(info):
        for key in frame_info.keys():
            repetition_variables[key].append(frame_info[key])
        for button_idx, button in enumerate(actions):
            repetition_variables[button].append(keys[frame_idx][button_idx])

    return repetition_variables


def make_gif(selected_frames, movie_fname):
    """Create a GIF file from a list of frames."""
    frame_list = [Image.fromarray(np.uint8(img), "RGB") for img in selected_frames]

    if not frame_list:
        logging.warning(f"No frames to save in {movie_fname}")
        return

    frame_list[0].save(
        movie_fname, save_all=True, append_images=frame_list[1:], optimize=False, duration=16, loop=0
    )


def make_mp4(selected_frames, movie_fname):
    """Create an MP4 file from a list of frames."""
    writer = skvideo.io.FFmpegWriter(
        movie_fname, inputdict={"-r": "60"}, outputdict={"-r": "60"}
    )
    for frame in selected_frames:
        im = Image.new("RGB", (frame.shape[1], frame.shape[0]), color="white")
        im.paste(Image.fromarray(frame), (0, 0))
        writer.writeFrame(np.array(im))
    writer.close()


def generate_savestate_from_frame(
    start_frame, bk2_fpath, output_fname, skip_first_step=True, game=None, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """Replay a bk2 file up to a specific frame and create a savestate."""
    replay = replay_bk2(
        bk2_fpath, skip_first_step=skip_first_step, game=game, scenario=scenario, inttype=inttype
    )
    for frame_idx, (_, _, _, _, _, state) in enumerate(replay):
        if frame_idx == start_frame:
            with gzip.open(output_fname, "wb") as fh:
                fh.write(state)
            break


def load_scenes_info(scenes_file):
    """Load scenes information from a CSV file."""
    scenes_info = pd.read_csv(scenes_file)
    scenes_info_dict = {}
    for idx, row in scenes_info.iterrows():
        try:
            scene_id = f'w{int(row["World"])}l{int(row["Level"])}s{int(row["Scene"])}'
            scenes_info_dict[scene_id] = {
                'start': int(row['Entry point']),
                'end': int(row['Exit point']),
                'level_layout': int(row['Layout'])
            }
        except:
            continue
    return scenes_info_dict


def process_bk2_file(bk2_info, args, scenes_info_dict, DERIVATIVES_FOLDER, STIMULI_PATH):
    """Process a single bk2 file to extract clips and savestates."""
    # Add stimuli path in each child process
    retro.data.Integrations.add_custom_path(STIMULI_PATH)

    error_logs = []
    processing_stats = {
        'bk2_file': bk2_info['bk2_file'],
        'clips_processed': 0,
        'clips_skipped': 0,
        'errors': 0,
    }

    try:
        bk2_file = bk2_info['bk2_file']
        bk2_idx = bk2_info['bk2_idx']
        sub = bk2_info['sub']
        ses = bk2_info['ses']
        run = bk2_info['run']
        skip_first_step = bk2_idx == 0

        logging.info(f"Processing bk2 file: {bk2_file}")
        rep_order_string = f'{str(ses).zfill(3)}{str(run).zfill(2)}{str(bk2_idx).zfill(2)}'
        curr_level = op.basename(bk2_file).split("_")[-2].split('-')[1]

        if curr_level in [x.split('s')[0] for x in scenes_info_dict.keys()]:
            repvars, frames_list = get_variables_from_replay(
                bk2_file, skip_first_step=skip_first_step, inttype=retro.data.Integrations.CUSTOM_ONLY
            )
            n_frames_total = len(frames_list)
            repvars['player_x_pos'] = [
                hi * 256 + lo for hi, lo in zip(repvars['player_x_posHi'], repvars['player_x_posLo'])
            ]

            # Look for clips
            scenes_in_current_level = [x for x in scenes_info_dict.keys() if curr_level in x]
            for current_scene in tqdm(scenes_in_current_level, desc=f"Processing scenes in {bk2_file}", leave=False):
                scenes_info_found = []
                scene_start = scenes_info_dict[current_scene]['start']
                scene_end = scenes_info_dict[current_scene]['end']
                level_layout = scenes_info_dict[current_scene]['level_layout']

                start_found = False
                for frame_idx in range(1, n_frames_total):
                    if not start_found:
                        if (
                            repvars['player_x_pos'][frame_idx] >= scene_start
                            and repvars['player_x_pos'][frame_idx - 1] < scene_start
                            and repvars['player_x_pos'][frame_idx] < scene_end
                            and repvars['level_layout'][frame_idx] == level_layout
                        ):
                            start_idx = frame_idx
                            start_found = True
                    else:
                        if (
                            (repvars['player_x_pos'][frame_idx] >= scene_end
                             and repvars['player_x_pos'][frame_idx - 1] < scene_end)
                            or (repvars['lives'][frame_idx] - repvars['lives'][frame_idx - 1] < 0)
                        ):
                            end_idx = frame_idx
                            start_found = False
                            scenes_info_found.append([start_idx, end_idx])
                        elif (
                            repvars['player_x_pos'][frame_idx] >= scene_start
                            and repvars['player_x_pos'][frame_idx - 1] < scene_start
                        ):
                            start_idx = frame_idx

                for pattern in scenes_info_found:
                    start_idx, end_idx = pattern
                    selected_frames = frames_list[start_idx:end_idx]
                    clip_code = f'{rep_order_string}{str(start_idx).zfill(7)}'
                    assert len(clip_code) == 14, f"Invalid clip code: {clip_code}"

                    # Construct BIDS-compliant paths
                    # Using 'mario_scenes' as the pipeline name
                    pipeline_name = 'mario_scenes'  # Name of the pipeline
                    deriv_folder = op.join(DERIVATIVES_FOLDER, pipeline_name)
                    sub_folder = op.join(deriv_folder, f"sub-{sub}")
                    ses_folder = op.join(sub_folder, f"ses-{ses}")
                    beh_folder = op.join(ses_folder, 'beh')
                    clips_folder = op.join(beh_folder, 'clips')
                    savestates_folder = op.join(beh_folder, 'savestates')
                    os.makedirs(clips_folder, exist_ok=True)
                    os.makedirs(savestates_folder, exist_ok=True)

                    # File names
                    entities = (
                        f"sub-{sub}_ses-{ses}_run-{run}_level-{repvars['level']}_"
                        f"scene-{int(current_scene.split('s')[1])}_clip-{clip_code}"
                    )

                    clip_fname = op.join(
                        clips_folder,
                        f"{entities}_beh.{args.clip_extension}",
                    )
                    savestate_fname = op.join(
                        savestates_folder,
                        f"{entities}_beh.state",
                    )

                    # Check if output files already exist
                    if op.exists(clip_fname) and op.exists(savestate_fname):
                        logging.info(f"Clip and savestate already exist for clip code {clip_code}, skipping.")
                        processing_stats['clips_skipped'] += 1
                        continue

                    try:
                        if args.clip_extension == 'gif':
                            make_gif(selected_frames, clip_fname)
                        elif args.clip_extension in ['mp3', 'mp4']:
                            make_mp4(selected_frames, clip_fname)
                        else:
                            raise ValueError(f"Unsupported clip extension: {args.clip_extension}")

                        # Save savestate
                        generate_savestate_from_frame(
                            start_idx, bk2_file, savestate_fname, skip_first_step=skip_first_step
                        )

                        # Save metadata as JSON sidecar files
                        metadata = {
                            'Subject': sub,
                            'Session': ses,
                            'Run': run,
                            'Level': repvars['level'],
                            'Scene': int(current_scene.split('s')[1]),
                            'ClipCode': clip_code,
                            'StartFrame': start_idx,
                            'EndFrame': end_idx,
                            'TotalFrames': n_frames_total,
                        }
                        metadata_fname = clip_fname.replace(f".{args.clip_extension}", ".json")
                        with open(metadata_fname, 'w') as json_file:
                            json.dump(metadata, json_file, indent=4)

                        processing_stats['clips_processed'] += 1
                    except Exception as e:
                        error_message = f"Error processing clip {clip_code} in bk2 file {bk2_file}: {str(e)}"
                        error_logs.append(error_message)
                        processing_stats['errors'] += 1
                        continue
    except Exception as e:
        error_message = f"Error processing bk2 file {bk2_file}: {str(e)}"
        error_logs.append(error_message)
        processing_stats['errors'] += 1
        print("Full traceback:")
        traceback.print_exc()
        
        # Optionally, print just the exception message
        print("\nError message:")
        print(e)

    return error_logs, processing_stats


def collect_bk2_files(DATA_PATH, subjects=None, sessions=None):
    """Collect all bk2 files and related information from the dataset."""
    bk2_files_info = []
    for root, _, files in sorted(os.walk(DATA_PATH)):
        if "sourcedata" not in root:
            for file in files:
                if "events.tsv" in file and "annotated" not in file:
                    run_events_file = op.join(root, file)
                    logging.info(f"Processing events file: {file}")
                    events_dataframe = pd.read_table(run_events_file)
                    events_dataframe = events_dataframe[events_dataframe['trial_type'] == 'gym-retro_game']
                    basename = op.basename(run_events_file)
                    entities = basename.split('_')
                    entities_dict = {}
                    for ent in entities:
                        if '-' in ent:
                            key, value = ent.split('-', 1)
                            entities_dict[key] = value
                    sub = entities_dict.get('sub')
                    ses = entities_dict.get('ses')
                    run = entities_dict.get('run')
                    if not sub or not ses or not run:
                        logging.warning(f"Could not extract subject, session, or run from filename {basename}")
                        continue
                    # Apply subject and session filters if specified
                    if subjects and sub not in subjects:
                        continue
                    if sessions and ses not in sessions:
                        continue
                    bk2_files = events_dataframe['stim_file'].values.tolist()
                    for bk2_idx, bk2_file in enumerate(bk2_files):
                        if bk2_file != "Missing file" and not isinstance(bk2_file, float):
                            bk2_files_info.append({
                                'bk2_file': bk2_file,
                                'bk2_idx': bk2_idx,
                                'sub': sub,
                                'ses': ses,
                                'run': run
                            })
    return bk2_files_info


def main(args):
    # Set up logging based on verbosity level
    if args.verbose == 0:
        logging_level = logging.WARNING
    elif args.verbose == 1:
        logging_level = logging.INFO
    elif args.verbose >= 2:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format='%(levelname)s: %(message)s')

    # Get datapath
    DATA_PATH = op.abspath(args.datapath)

    # Load scenes info
    SCENES_FILE = args.scenesfile
    if SCENES_FILE is None:
        SCENES_FILE = op.join(
            DATA_PATH, "code", "annotations", "scenes", "resources", "scenes_mastersheet.csv"
        )

    scenes_info_dict = load_scenes_info(SCENES_FILE)

    # Setup derivatives folder
    if args.output is None:
        DERIVATIVES_FOLDER = op.join(DATA_PATH, "derivatives")
    else:
        DERIVATIVES_FOLDER = op.abspath(args.output)
    os.makedirs(DERIVATIVES_FOLDER, exist_ok=True)

    # Integrate game
    if args.stimuli_path is None:
        STIMULI_PATH = op.abspath(op.join(DATA_PATH, "stimuli"))
    else:
        STIMULI_PATH = op.abspath(args.stimuli_path)
    logging.debug(f"Adding custom stimuli path: {STIMULI_PATH}")
    retro.data.Integrations.add_custom_path(STIMULI_PATH)
    games_list = retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY)
    logging.debug(f"Available games: {games_list}")

    logging.info(f"Generating annotations for the Mario dataset in: {DATA_PATH}")
    logging.info(f"Taking stimuli from: {STIMULI_PATH}")
    logging.info(f"Saving derivatives in: {DERIVATIVES_FOLDER}")
    logging.info(f"Using scenes file: {SCENES_FILE}")

    # Collect all bk2 files and related information
    bk2_files_info = collect_bk2_files(DATA_PATH, args.subjects, args.sessions)
    total_bk2_files = len(bk2_files_info)

    # Process bk2 files in parallel with progress bar
    n_jobs = args.n_jobs
    logging.info(f"Processing {total_bk2_files} bk2 files using {n_jobs} job(s)...")

    with tqdm_joblib(tqdm(desc="Processing bk2 files", total=total_bk2_files)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_bk2_file)(bk2_info, args, scenes_info_dict, DERIVATIVES_FOLDER, STIMULI_PATH)
            for bk2_info in bk2_files_info
        )

    # Initialize aggregators
    total_processing_stats = {
        'total_bk2_files': total_bk2_files,
        'total_clips_processed': 0,
        'total_clips_skipped': 0,
        'total_errors': 0,
    }
    all_error_logs = []

    # Process results
    for error_logs, processing_stats in results:
        total_processing_stats['total_clips_processed'] += processing_stats.get('clips_processed', 0)
        total_processing_stats['total_clips_skipped'] += processing_stats.get('clips_skipped', 0)
        total_processing_stats['total_errors'] += processing_stats.get('errors', 0)
        all_error_logs.extend(error_logs)

    # Prepare data for saving
    # Save dataset description as per BIDS derivatives
    dataset_description = {
        'Name': 'Mario Scenes',
        'BIDSVersion': '1.6.0',
        'GeneratedBy': [{
            'Name': 'Courtois Neuromod',
            'Version': '1.0.0',
            'CodeURL': 'https://github.com/courtois-neuromod/mario_scenes/script/clip_extractor.py'  # Update with actual URL
        }],
        'SourceDatasets': [{'URL': 'n/a'}],
        'License': 'CC0',
    }
    deriv_folder = op.join(DERIVATIVES_FOLDER, 'mario_scenes')
    os.makedirs(deriv_folder, exist_ok=True)
    with open(op.join(deriv_folder, "dataset_description.json"), "w") as f:
        json.dump(dataset_description, f, indent=4)

    # Write error logs to a log file
    log_file = op.join(deriv_folder, "processing_log.txt")
    with open(log_file, "w") as f:
        f.write("Processing Log\n")
        f.write("=================\n")
        f.write(f"Total bk2 files: {total_processing_stats['total_bk2_files']}\n")
        f.write(f"Total clips processed: {total_processing_stats['total_clips_processed']}\n")
        f.write(f"Total clips skipped: {total_processing_stats['total_clips_skipped']}\n")
        f.write(f"Total errors: {total_processing_stats['total_errors']}\n")
        f.write("\nError Details:\n")
        for error in all_error_logs:
            f.write(error + "\n")
    logging.info(f"Processing complete. Log file saved to {log_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clips from Mario dataset based on scene information.")
    parser.add_argument(
        "-d",
        "--datapath",
        default='.',
        type=str,
        help="Data path to look for events.tsv and .bk2 files. Should be the root of the Mario dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Path to the derivatives folder, where the outputs will be saved.",
    )
    parser.add_argument(
        "-s",
        "--scenesfile",
        default=None,
        type=str,
        help="Path to the scenes file, a CSV file that contains info about the start and end positions to clip.",
    )
    parser.add_argument(
        "-sp",
        "--stimuli_path",
        default=None,
        type=str,
        help="Path to the stimuli folder containing the game ROMs. Defaults to <datapath>/stimuli if not specified.",
    )
    parser.add_argument(
        "-ext",
        "--clip_extension",
        default="gif",
        type=str,
        help="Format in which the extracted clips should be saved.",
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        default=1,
        type=int,
        help="Number of CPU cores to use for parallel processing.",
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Increase verbosity level (can be specified multiple times)'
    )
    parser.add_argument(
        '--subjects', '-sub', nargs='+', default=None,
        help='List of subjects to process (e.g., sub-01 sub-02). If not specified, all subjects are processed.'
    )
    parser.add_argument(
        '--sessions', '-ses', nargs='+', default=None,
        help='List of sessions to process (e.g., ses-001 ses-002). If not specified, all sessions are processed.'
    )

    args = parser.parse_args()
    main(args)
