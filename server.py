import gradio as gr
import argparse
import tempfile
import torch
import whisper
import os
import subprocess
from whisper.utils import get_writer

from languages import LANGUAGES
import ytdlp_functions

FULL_TO_CODE = {v: k for k, v in LANGUAGES.items()}


def download_video(url, quick, language, model, task, addSrtToVideo):
    if quick:
        returned_yt_file = ytdlp_functions.download_quick_mp4(url=url, folder=str(tempfile.gettempdir()))

        return transcribe(
            inputFile=returned_yt_file,
            language=language,
            model=model,
            task=task,
            addSrtToVideo=addSrtToVideo)


def transcribe(inputFile, language, model, task, addSrtToVideo):
    print("gpu available: " + str(torch.cuda.is_available()))
    gpu = torch.cuda.is_available()
    model = whisper.load_model(model)
    if inputFile is None:
        raise gr.Error("No input file provided")
    # ytdlp_functions will give us a string, gradio filepicker an actual file
    inputFileCleared = inputFile if isinstance(inputFile, str) else inputFile.name

    whisperOutput = model.transcribe(
        inputFileCleared,
        task=task,
        language=FULL_TO_CODE[language],
        verbose=True,
        fp16=gpu
    )

    temp_dir = str(tempfile.gettempdir())
    writer = get_writer("srt", temp_dir)
    writer(whisperOutput, inputFileCleared)

    # Construct the correct srtFile path
    srtFile = os.path.join(temp_dir, os.path.basename(inputFileCleared).rsplit(".", 1)[0] + ".srt")

    # broken srt filepaths. Use those if tempdir above acts weird.
    # srtFile = f"{inputFileCleared}" + ".srt"
    # anotherSrtFile = inputFileCleared.rsplit(".", 2)[0] + ".srt"
    # srtFile = inputFileCleared.rsplit(".", 1)[0] + ".srt"
    if addSrtToVideo:
        video_out = inputFileCleared + "_output.mkv"

        try:
            command = [
                'ffmpeg',
                '-i', inputFileCleared,
                '-vf', f'subtitles={srtFile}',
                '-c:a', 'copy',
                video_out
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            raise gr.Error("ffmpeg failed to embed subtitles into video")

        return video_out

    return srtFile



with gr.Blocks() as app:
    gr.Markdown("# whisper-subtitles-webui")
    with gr.Tab("Subtitle Video"):
        st_file = gr.File()
        st_lang = gr.Dropdown(
            label="Language",
            choices=list(LANGUAGES.values()),
            value=LANGUAGES["en"]
        )
        st_model = gr.Dropdown(["tiny", "small", "medium", "large", ], label="Model", value="tiny")
        st_task = gr.Radio(["transcribe", "translate"], label="Task", value="transcribe")
        st_embed = gr.Checkbox(label="embed subtitles into video file (ffmpeg required)")
        st_file_out = gr.File()
        st_start_button = gr.Button("Run", variant="primary")
    with gr.Tab("YouTube to Subtitle (experimental)"):
        gr.Markdown(">try to update yt-dlp if downloads don't work")
        yt_url = gr.Textbox(label="YouTube URL", placeholder="YouTube URL")
        yt_quick = gr.Checkbox(label="Quick settings", value=True, interactive=False)
        yt_lang = gr.Dropdown(
            label="Language",
            choices=list(LANGUAGES.values()),
            value=LANGUAGES["en"]
        )
        yt_model = gr.Dropdown(["tiny", "small", "medium", "large"], label="Model", value="tiny")
        yt_task = gr.Radio(["transcribe", "translate"], label="Task", value="transcribe")
        yt_embed = gr.Checkbox(label="embed subtitles into video file (ffmpeg required)")
        yt_file_out = gr.File()
        yt_start_button = gr.Button("Download and Run", variant="primary")

    st_start_button.click(fn=transcribe, inputs=
    [st_file,
     st_lang,
     st_model,
     st_task,
     st_embed
     ], outputs=st_file_out, api_name="video_to_subs")
    yt_start_button.click(fn=download_video, inputs=
    [
        yt_url,
        yt_quick,
        yt_lang,
        yt_model,
        yt_task,
        yt_embed,
    ], outputs=yt_file_out, api_name="yt_to_subs")
parser = argparse.ArgumentParser(description='Share option')
parser.add_argument('--remote', type=bool, help='share', default=False)
args = parser.parse_args()
app.launch(share=args.remote)
