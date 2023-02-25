from yt_dlp import YoutubeDL
import os

def download_quick_mp4(url, folder):
    ytdl_format_options = {
        'outtmpl': os.path.join(folder, '%(title)s-%(id)s.%(ext)s')
    }

    with YoutubeDL(ytdl_format_options) as ydl:
        info = ydl.extract_info(url, download=False)
        file = ydl.download(url)

        return os.path.join(folder, f"{info['title']}-{info['id']}.{info['ext']}")