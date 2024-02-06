import matplotlib.font_manager
import matplotlib.pyplot as plt  # noqa: F401
from tempfile import NamedTemporaryFile
from fontTools import ttLib
import re
import requests


def get_google_font(fontname):
    api_fontname = fontname.replace(' ', '+')

    api_response = requests.get(
        f"https://fonts.googleapis.com/css?family={api_fontname}:black,bold,regular,light"
    )

    font_urls = re.findall(r'(https?://[^\)]+)', str(api_response.content))

    for font_url in font_urls:
        font_data = requests.get(font_url)
        f = NamedTemporaryFile(delete=False, suffix='.ttf')
        f.write(font_data.content)
        f.close()
        font = ttLib.TTFont(f.name)
        font_family_name = font['name'].getDebugName(1)
        matplotlib.font_manager.fontManager.addfont(f.name)
        print(f"Added new font as {font_family_name}")
