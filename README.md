[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EliasVincent/whisper-subtitles-webui/blob/master/colab/whisper_subtitles_webui_colab.ipynb)

now with Whisper Large v3!

# What

A gradio frontend for generating transcribed or translated subtitles for videos using OpenAI Whisper locally.

# Why

I just wanted a nice frontend where you can just drop a video or url and it will spit out subs. Whisper is amazing but I haven't found that many implementations, especially ones that can be run locally.

![](img/1.png)

# Install

```
python -m venv .
.\Scripts\activate # windows
source bin/activate # mac/linux


# if this doesn't work, pip install the following manually: openai-whisper ffmpeg torch gradio yt-dlp
pip install -r requirements.txt

python server.py
```

To network-share the instance, add `--remote=True`.

# Features

- Input a video or any other media file
- Input a YouTube URL
- Transcribe
- Translate to English
- Select different models for your hardware
- CUDA support
- Output .srt or video file with embedded subtitles

# Troubleshooting

If the output says `gpu available: False` [you might need to pip install a different version of Torch for your specific hardware](https://pytorch.org/get-started/locally/#start-locally)

You might need to have ffmpeg installed on your system if you have issues with the included. [Download here](https://ffmpeg.org/download.html). Make sure it's working when invoking `ffmpeg` from the command line.
