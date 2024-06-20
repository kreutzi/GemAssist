from yt_dlp import YoutubeDL

YDL_OPTIONS = { 'format': 'bestaudio', 'noplaylist': 'True','extract_flat':True }

def search_youtube(query):
    with YoutubeDL(YDL_OPTIONS) as ydl:
        try:
            info = ydl.extract_info(f"ytsearch5:{query}", download=False)
            entries = info['entries'] # Get the first 10 entries
            urls = [[entry['title'],entry['url']]for entry in entries]
            return urls
        except Exception as e:
            print(e)
            return None
        
