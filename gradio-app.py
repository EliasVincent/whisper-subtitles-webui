import gradio as gr
import ffmpeg
import os
import tempfile
import torch
import whisper
from whisper.utils import get_writer

def transcribe(inputFile, language, model, task, addSrtToVideo):
    print("gpu available: " + str(torch.cuda.is_available()))
    gpu = torch.cuda.is_available()
    model = whisper.load_model(model)
    whisperOutput = model.transcribe(
        inputFile.name, 
        task=task, 
        language=language, 
        verbose=True,
        fp16=gpu
    )
    
    writer = get_writer("srt", str(tempfile.gettempdir()))
    writer(whisperOutput, inputFile.name)
    srtFile = f"{inputFile.name}" + ".srt"
    
    if addSrtToVideo:
        video_out = inputFile.name + "_output.mkv"

        input_ffmpeg = ffmpeg.input(inputFile.name)
        input_ffmpeg_sub = ffmpeg.input(srtFile)

        input_video = input_ffmpeg['v']
        input_audio = input_ffmpeg['a']
        input_subtitles = input_ffmpeg_sub['s']
        stream = ffmpeg.output(
            input_video, input_audio, input_subtitles, video_out,
            vcodec='copy', acodec='copy', scodec='srt'
        )
        stream = ffmpeg.overwrite_output(stream)
        #execute(stream, desc=f"Adding `SubRip Subtitle file` to {inputFile.name}")
        ffmpeg.run(stream)
        return video_out
    
    return srtFile

demo = gr.Interface(
    fn=transcribe,
    inputs=[
        "file",
        gr.Textbox(value="source language (en, de, ja, ..)"),
        gr.Dropdown(["tiny", "small", "medium", "large"], value="tiny"), 
        gr.Radio(["transcribe", "translate"], value="translate"),
        gr.Checkbox(label="embed subtitles into video file")
    ],
    outputs="file"
)

demo.launch()
