# from downloader import download

from database.database import TOPIC_ENUM
from bing_image_downloader.downloader import download




topic = TOPIC_ENUM.fireproof_paper.value
download(topic, topic=topic, output_dir="./images/", limit=1000, timeout=1)
