from flickrapi import FlickrAPI
from urllib.request import urlretrieve
# from pprint import pprint
import os, time, sys

key = ""
secret = ""
wait_time = 1

animal_name = sys.argv[1]
save_dir = "./" + animal_name

# flickrAPI
flickr = FlickrAPI(key, secret, format="parsed-json")
result = flickr.photos.search(
    text = animal_name,          # 検索ワード
    per_page = 400,             # 取得件数
    media = 'photos',           # メディアの種類
    sort = 'relevance',         # 順番,関連順
    safe_seach = 1,             # セーフサーチ
    extras = 'url_q, licence'   # 返り値
)

photos = result['photos']
pprint(photos)


for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = save_dir + '/' + photo['id'] + '.jpg'

    # ファイルが重複してたら飛ばす
    if os.path.exists(filepath): continue

    urlretrieve(url_q, filepath) # ダウンロードを実行.
    time.sleep(wait_time)
