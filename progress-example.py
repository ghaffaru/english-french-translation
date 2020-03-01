import requests

from clint.textui import progress

import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
url = 'https://www.floydhub.com/api/v1/resources/Av2ThePYtAHXMAuSXEBV8X/glove.6B.100d.txt?content=true&rename=glove6b100dtxt'

r = requests.get(url, stream=True)

with open(os.path.join(BASE_DIR, 'data/glove.6B.100d.txt'), 'wb') as file:

        raw_content = r.raw.read()
        total_length = len(raw_content)

        for ch in progress.bar(r.iter_content(chunk_size = 2391975), expected_size=(total_length/1024) + 1):

                if ch:

                        file.write(ch)
