import gradio as gr
import ffmpeg as ff
import os
import tempfile
import torch
import whisper
from whisper.utils import get_writer

def transcribe(inputFile, language, model, task):
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
    
    return srtFile

demo = gr.Interface(
    fn=transcribe,
    inputs=[
        "file",
        gr.Textbox(value="source language (en, de, ja, ..)"),
        gr.Dropdown(["tiny", "small", "medium", "large"], value="tiny"), 
        gr.Radio(["transcribe", "translate"], value="translate")
    ],
    outputs="file"
)

demo.launch()
