from __future__ import unicode_literals
import yt_dlp as youtube_dl
print(youtube_dl)

def get_audio(video):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'output',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }
         ],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video])

if __name__ == "__main__":
    get_audio('https://www.youtube.com/watch?v=4HgShra-KnY')
