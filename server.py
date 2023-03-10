import gradio as gr
import ffmpeg
import argparse
import tempfile
import torch
import whisper
from whisper.utils import get_writer

import ytdlp_functions

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
    # ytdlp_functions will give us a string, gradio filepicker an actual file
    inputFileCleared = inputFile if isinstance(inputFile, str) else inputFile.name
    
    whisperOutput = model.transcribe(
        inputFileCleared, 
        task=task, 
        language=language, 
        verbose=True,
        fp16=gpu
    )
    
    writer = get_writer("srt", str(tempfile.gettempdir()))
    writer(whisperOutput, inputFileCleared)
    srtFile = f"{inputFileCleared}" + ".srt"
    
    if addSrtToVideo:
        video_out = inputFileCleared + "_output.mkv"

        input_ffmpeg = ffmpeg.input(inputFileCleared)
        input_ffmpeg_sub = ffmpeg.input(srtFile)

        input_video = input_ffmpeg['v']
        input_audio = input_ffmpeg['a']
        input_subtitles = input_ffmpeg_sub['s']
        stream = ffmpeg.output(
            input_video, input_audio, input_subtitles, video_out,
            vcodec='copy', acodec='copy', scodec='srt'
        )
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)
        return video_out
    
    return srtFile

with gr.Blocks() as app:
    gr.Markdown("# whisper-subtitles-webui")
    with gr.Tab("Subtitle Video"):
        st_file = gr.File()
        st_lang = gr.Textbox(label="Language", placeholder="source language (en, de, ja, ..)")
        st_model = gr.Dropdown(["tiny", "small", "medium", "large",], label="Model", value="tiny")
        st_task = gr.Radio(["transcribe", "translate"], label="Task", value="translate")
        st_embed = gr.Checkbox(label="embed subtitles into video file")
        st_file_out = gr.File()
        st_start_button = gr.Button("Run", variant="primary")
    with gr.Tab("YouTube to Subtitle"):
        gr.Markdown(">try to update yt-dlp if downloads don't work")
        yt_url = gr.Textbox(placeholder="YouTube URL")
        yt_quick = gr.Checkbox(label="quick settings", value=True, interactive=False)
        yt_lang = gr.Textbox(label="Language", placeholder="source language (en, de, ja, ..)")
        yt_model = gr.Dropdown(["tiny", "small", "medium", "large"], label="Model", value="tiny")
        yt_task = gr.Radio(["transcribe", "translate"], label="Task", value="translate")
        yt_embed = gr.Checkbox(label="embed subtitles into video file")
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
