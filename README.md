[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EliasVincent/whisper-subtitles-webui/blob/master/colab/whisper_subtitles_webui_colab.ipynb)

A gradio frontend for generating transcribed or translated subtitles for videos using OpenAI Whisper locally.

![](img/1.png)

# Install

This has been tested with Python 3.12

```
python -m venv .venv
```

Activate the venv on Windows with

```
.\.venv\Scripts\activate
```

On Linux and Mac, run

```
source .venv/bin/activate
```

Now, install the pip dependencies:

```
# if this doesn't work, pip install the following manually: openai-whisper ffmpeg torch gradio

pip install -r requirements.txt
```

Then launch the server with

```
python server.py
```

To share, add `--remote=True`.

For embedding subtitles into a video, you need to have `ffmpeg` installed on your system.

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

# Why

I just wanted a nice frontend where you can just drop a video or url and it will spit out subs. Whisper is amazing but I haven't found that many implementations, especially ones that can be run locally.

# Privacy

By default, Gradio sends some data to its servers. Gradio provides some variables to disable them, which have been applied here.
