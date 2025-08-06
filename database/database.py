import os
import subprocess
import time
import random
import shutil
from datetime import datetime
from typing import List
import yaml
import pymongo
from tqdm import tqdm
from pymongo.collection import Collection
from enum import Enum
from persiantools.jdatetime import JalaliDate
import urllib.parse
import pandas as pd
from dotenv import load_dotenv
import re

load_dotnev()


class COLLECTION_ENUM(Enum):
    automate_news = "automate_news"
    start_title = "start_title"
    end_title = "end_title"
    images = "images"
    summary = "summary"
    text = "text"
    monthly = "monthly"
    auth = "auth"
    topics = "topics"
    article_doi = "article_doi"
    wrong_words = "wrong_words"


class TOPIC_ENUM(Enum):
    # emtedad
    sound_insulation = "sound insulation"
    fireclays = "fireclays"
    fireproof_blanket = "fireproof blanket"
    fireproof_rope = "fireproof rope"
    fireproof_paper = "fireproof paper"

    # lia filter
    bag_filter = "bag filter"
    carbon_filter = "carbon filter"
    dust_filter = "dust filter"


class Database:
    def __init__(self) -> None:
        self.mongo_client = self.connect_to_db()
        self.update_translation = True
        self.init_database()
        # print("Remaining Text Documents are:", self.remaining_text())

    def connect_to_db(self) -> pymongo.MongoClient:

        username = urllib.parse.quote_plus(os.getenv("MONGO_INITDB_ROOT_USERNAME"))
        password = urllib.parse.quote_plus(os.getenv("MONGO_INITDB_ROOT_PASSWORD"))
        host = os.getenv("MONGO_HOST")
        port = os.getenv("MONOG_PORT")
        mongo_client: pymongo.MongoClient = pymongo.MongoClient(
            f"mongodb://{username}:{password}@{host}:{port}"
        )
        return mongo_client

    def init_database(self):
        self.automate_news_db = self.mongo_client[COLLECTION_ENUM.automate_news.value]
        self.start_title_col = self.automate_news_db[COLLECTION_ENUM.start_title.value]
        self.end_title_col = self.automate_news_db[COLLECTION_ENUM.end_title.value]
        self.images_col = self.automate_news_db[COLLECTION_ENUM.images.value]
        self.summary_col = self.automate_news_db[COLLECTION_ENUM.summary.value]
        self.text_col = self.automate_news_db[COLLECTION_ENUM.text.value]
        self.monthly_col = self.automate_news_db[COLLECTION_ENUM.monthly.value]
        self.auth_col = self.automate_news_db[COLLECTION_ENUM.auth.value]
        self.topics_col = self.automate_news_db[COLLECTION_ENUM.topics.value]

        start_title_df = pd.read_csv("./database/start_title.csv", index_col="index")
        for index, row in start_title_df.iterrows():
            key = {"key": index}
            data = {"name": row["name"]}
            self.start_title_col.update_one(key, {"$set": data}, upsert=True)

        to_add_data = [
            ("./database/end_title_sound_insulation.csv", TOPIC_ENUM.sound_insulation.value),
            ("./database/end_title_fireclays.csv", TOPIC_ENUM.fireclays.value),
            ("./database/end_title_fireproof_blanket.csv", TOPIC_ENUM.fireproof_blanket.value),
            ("./database/end_title_fireproof_rope.csv", TOPIC_ENUM.fireproof_rope.value),
            ("./database/end_title_fireproof_paper.csv", TOPIC_ENUM.fireproof_paper.value),
            ("./database/end_title_bag_filter.csv", TOPIC_ENUM.bag_filter.value),
            ("./database/end_title_carbon_filter.csv", TOPIC_ENUM.carbon_filter.value),
            ("./database/end_title_dust_filter.csv", TOPIC_ENUM.dust_filter.value),
        ]

        for file_path, topic_name in to_add_data:
            self.update_end_title_df(file_path, topic_name)

        # for manually adding text from txt files
        # text_list = (
        #     ("./data/text_sound_insulation.txt", TOPIC_ENUM.sound_insulation.value),
        #     ("./data/text_fireclays.txt", TOPIC_ENUM.fireclays.value),
        # )
        # for text_file, topic in text_list:
        #     self.add_text_to_db(text_file=text_file, topic=topic)

        self.update_summary_col()
        self.update_auth_collection()
        self.update_topics_collection()
        self.set_up_article_doi_col()
        self.set_up_wrong_words_col()
        # self.update_text_col_from_article()

        if self.update_translation:
            self.batch_translate_with_crow(count=1000)

    def update_summary_col(self):
        summary_df = pd.read_csv("./database/summary.csv", index_col="index")
        for index, row in summary_df.iterrows():
            key = {"key": index}
            data = {"name": row["summary"], "title_id": row["title_id"]}
            self.summary_col.update_one(key, {"$set": data}, upsert=True)

    def set_up_wrong_words_col(self):
        self.wrong_words_col = self.automate_news_db[COLLECTION_ENUM.wrong_words.value]
        self.wrong_wrong_index_name = "wrong_words_key_index"
        self.create_index(
            self.wrong_words_col, "wrong_word", self.wrong_wrong_index_name
        )

    def add_word_to_wrong_word_col(self, wrong_word, nltk_result, topic):
        key = {"wrong_word": wrong_word.lower()}

        data = {
            "$inc": {"count": 1, f"count_{topic}": 1},
            # "$addToSet": {"words": wrong_word},
            "$addToSet": {"topics": topic, "words": wrong_word},
            # "nltk_result": nltk_result,
        }
        if self.wrong_words_col.count_documents(key) == 0 and len(nltk_result) > 0:
            data["$set"] = {"nltk_result": nltk_result}

        self.wrong_words_col.update_one(
            key,
            data,
            upsert=True,
        )

    def set_up_article_doi_col(self):
        self.article_doi_col = self.automate_news_db[COLLECTION_ENUM.article_doi.value]
        # create article doi index with unique key
        self.article_doi_index_name = "article_doi_key_index"
        self.create_index(self.article_doi_col, "doi", self.article_doi_index_name)

    def add_article_doi(self, topic, doi):
        # check doi exist, if not create it
        key = {"doi": doi}
        doi_doc = self.article_doi_col.find_one(key)
        if not doi_doc:
            data = {
                "doi": doi,
                "topic": topic,
                "date_added": datetime.now(),
                "used": False,
                "download": "not",
                "file_name": "",
                "sent": False,
                "num_pages": 0,
                "is_text": False,
                "text": "",
                "text_date": None,
                "text_processed": False,
            }

            self.article_doi_col.insert_one(data)
        else:
            print("Article with doi", doi, "exist")

    def get_random_article_doi(self, topic):
        doi = self.get_random_article(topic=topic)
        return doi

    def get_random_article(self, topic):
        article_doi_doc = self.article_doi_col.aggregate(
            [
                {"$match": {"topic": topic, "used": False, "download": "not"}},
                {"$sample": {"size": 1}},
            ]
        )
        article_doi_doc_list = list(article_doi_doc)
        if len(article_doi_doc_list) > 0:
            return article_doi_doc_list[0]
        else:
            return None

    def reset_downloading_article(self):
        self.article_doi_col.update_many(
            {"download": "downloading"}, {"$set": {"download": "not", "used": False}}
        )

    def get_number_of_not_downloaded_article(self):
        return self.article_doi_col.count_documents({"download": "not"})

    def mark_article_as_downloading(self, doi):
        key = {"doi": doi}
        data = {"download": "downloading"}
        self.article_doi_col.update_one(key, {"$set": data}, upsert=True)

    def get_article_from_doi(self, doi):
        key = {"doi": doi}
        article_doi_doc = self.article_doi_col.find_one(key)
        return article_doi_doc

    def is_any_article_downloading(self):
        article_doi_doc = self.article_doi_col.find_one({"download": "downloading"})
        if article_doi_doc:
            return True
        else:
            return False

    def mark_article_as_downloaded(self, file_name, num_pages):
        article_doi_doc = self.article_doi_col.update_one(
            {"download": "downloading"},
            {
                "$set": {
                    "download": "downloaded",
                    "used": True,
                    "file_name": file_name,
                    "num_pages": num_pages,
                }
            },
        )

    def get_list_of_article_doi(self):
        article_doi_doc = self.article_doi_col.find()
        article_doi_doc = list(article_doi_doc)
        if len(article_doi_doc) > 0:
            return article_doi_doc
        else:
            return None

    def mark_article_as_used(self, doi):
        key = {"doi": doi}
        data = {"used": True, "download": "downloaded"}
        self.article_doi_col.update_one(key, {"$set": data})

    def get_random_article_doi_to_send(self, topic=None):
        query = {
            "used": True,
            "download": "downloaded",
            "sent": False,
            "num_pages": {"$gt": 0},
            "file_name": {"$ne": "", "$ne": "not_available"},
        }
        if topic is not None:
            query["topic"] = topic

        article_doi = self.article_doi_col.aggregate(
            [{"$match": query}, {"$sample": {"size": 1}}]
        )
        if article_doi:
            article_doi = list(article_doi)
            if len(article_doi) > 0:
                return article_doi[0]
        return None

    def mark_article_as_sent(self, doi):
        key = {"doi": doi}
        data = {"sent": True}
        self.article_doi_col.update_one(key, {"$set": data})

    def save_article_txt_from_file(self, doi, txt_file):
        article_doi = self.article_doi_col.find_one({"doi": doi})
        new_cost = 0
        if article_doi is not None:
            with open(txt_file, "r") as f:
                text = f.read()

                self.article_doi_col.update_one(
                    {"doi": doi},
                    {
                        "$set": {
                            "text": text,
                            "is_text": True,
                            "text_date": datetime.now(),
                        }
                    },
                )
                new_cost = self.update_monthly_spend(article_doi["num_pages"])

        os.remove(txt_file)
        return new_cost

    def update_end_title_df(self, filename, topic):
        end_title_df = pd.read_csv(filename, index_col="index")
        for index, row in end_title_df.iterrows():
            key = {"key": index, "topic": topic}
            data = {"name": row["name"]}
            self.end_title_col.update_one(key, {"$set": data}, upsert=True)

    def get_new_text_key(self):
        item = self.text_col.aggregate(
            [{"$group": {"_id": "$item", "key": {"$max": "$key"}}}]
        )
        return list(item)[0]["key"] + 1

    def ger_random_doc(self, col: Collection, topic=None) -> str:
        if topic:
            doc = col.aggregate(
                [{"$match": {"topic": topic}}, {"$sample": {"size": 1}}]
            )
        else:
            doc = col.aggregate([{"$sample": {"size": 1}}])
        doc = list(doc)
        if len(doc) != 0:
            return doc[0]["name"]
        else:
            raise Exception("No Doc Found")

    def get_random_start_title(self) -> str:
        name = self.ger_random_doc(self.start_title_col)
        return name

    def get_random_end_title(self, topic) -> str:
        name = self.ger_random_doc(self.end_title_col, topic=topic)
        return name

    def get_random_image_url(self, topic) -> str:
        image_url = self.ger_random_doc(self.images_col, topic=topic)
        return image_url

    def get_summary(self, title_name):
        title_doc = self.start_title_col.find_one({"name": title_name})
        if title_doc:
            summary = self.summary_col.find_one({"title_id": title_doc["key"]})
            if summary:
                return summary["name"]
        return None

    def get_random_text(self, topic):
        doc = self.text_col.aggregate(
            [
                {
                    "$match": {
                        "used": False,
                        "topic": topic,
                    }
                },
                {"$sample": {"size": 1}},
            ]
        )
        doc = list(doc)
        doc = doc[0]
        if doc:
            return doc
        else:
            raise Exception("No Text Found")

    def save_persian_text(self, text_id, persian_text):
        key = {"_id": text_id}
        data = {"persian_text": persian_text}
        self.text_col.update_one(key, {"$set": data}, upsert=True)

    def has_persian_text(self, text_id):
        key = {"_id": text_id, "persian_text": {"$exists": True}}
        data = self.text_col.find_one(key)
        if data:
            persian_text = data.get("persian_text", None)
            if persian_text is None or persian_text == "":
                return False
            return True
        else:
            return False

    def translate_with_crow(self, text):
            command = [
                "flatpak",
                "run",
                "io.crow_translate.CrowTranslate",
                "-s",
                "en",
                "-t",
                "fa",
                "--brief",
                text,
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE)
            persian_text = result.stdout
            persian_text = (
                persian_text.decode("unicode-escape").encode("latin1").decode("utf-8")
            )
            persian_text = str(persian_text)
            if len(persian_text) > 1:
                return persian_text
            raise Exception("Can't translate")

    def batch_translate_with_crow(self, topic=None, count=100):
        # all_text = d.text_col.find({})
        all_text = self.text_col.aggregate(
            [
                {
                    "$match": {
                        "$or": [
                            {"persina_text": {"$eq": ""}},
                            {"persian_text": {"$exists": False}},
                        ],
                        # "used": {"$eq": False}
                    }
                },
                {"$limit": count},
            ]
        )
        for c, text_doc in zip(range(count), all_text):
            object_id = text_doc["_id"]

            has_persina_text = self.has_persian_text(object_id)
            print(f"[{c}/{count}], {object_id} | has: {has_persina_text}")
            if not has_persina_text:
                text = text_doc["text"]
                persian_text = self.translate_with_crow(text)
                # print("output:", persian_text)
                self.save_persian_text(object_id, persian_text)
            time.sleep(random.randint(1, 10))

    def mark_text_as_done(self, text_id):
        key = {"_id": text_id}
        data = {"used": True}
        self.text_col.update_one(key, {"$set": data}, upsert=True)

    def add_one_to_monthly(self, topic):
        month = f"{JalaliDate.today().year}-{JalaliDate.today().month}"
        current_mont_doc = self.monthly_col.find_one({"month": month})
        if current_mont_doc:
            self.monthly_col.update_one(
                {"month": month}, { "$inc": {topic: 1}}
            )
            self.monthly_col.update_one(
                {"month": month}, { "$inc": {"count": 1}}
            )
        else:
            topics_dict = {}
            for topic_enum in TOPIC_ENUM:
                if topic == topic_enum.value:
                    topics_dict[topic_enum.value] = 1
                else:
                    topics_dict[topic_enum.value] = 0
            self.monthly_col.insert_one({"month": month, "count": 1, **topics_dict})

    def update_monthly_spend(self, num_pages):
        new_cost = 2500
        if 10 <= num_pages < 15:
            new_cost = 5000
        elif 15 <= num_pages < 20:
            new_cost = 10000
        elif num_pages >= 20:
            new_cost = 15000
        elif num_pages >= 25:
            new_cost = 15000 + (num_pages - 25) * 200

        month = f"{JalaliDate.today().year}-{JalaliDate.today().month}"
        # current_mont_doc = self.monthly_col.find_one(
        #     {"month": month})
        self.monthly_col.update_one(
            {"month": month},
            {"$inc": {"cost": new_cost}},
            upsert=True,
        )
        return new_cost

    def get_monthly_spend(self):
        month = f"{JalaliDate.today().year}-{JalaliDate.today().month}"
        month_doc = self.monthly_col.find_one({"month": month})
        sum_of_cost = self.monthly_col.aggregate(
            [{"$group": {"_id": None, "sum": {"$sum": "$cost"}}}]
        )
        if month_doc is not None:
            return month_doc["cost"], list(sum_of_cost)[0]["sum"]
        return 0, 0

    def remaining_text(self, topic) -> int:
        count = self.text_col.count_documents({"topic": topic, "used": False})
        return count

    def create_index(self, col, key, name):
        if name not in col.index_information():
            col.create_index(key, unique=True, name=name)

    def update_auth_collection(self):
        # create auth key index with unique key
        self.auth_key_index_name = "auth_key_index"
        self.create_index(self.auth_col, "key", self.auth_key_index_name)

        # update auth collection with user and password from yaml file
        with open("./database/user_pass.yaml", "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            for key, value in config.items():
                key = {"key": key}
                data = {
                    "email": value["email"],
                    "password": value["password"],
                    "topics": value["topics"],
                }
                self.auth_col.update_one(key, {"$set": data}, upsert=True)

    def update_topics_collection(self):
        self.topics_key_index_name = "topics_key_index"
        self.create_index(self.topics_col, "key", self.topics_key_index_name)

        # update the topics collection from topics yaml file
        with open("./database/topics_list.yaml", "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            for key, value in config.items():
                key = {"key": key}
                data = {"names": value["names"], "count": value["count"]}
                self.topics_col.update_one(key, {"$set": data}, upsert=True)

    def get_auth_from_collection(self):
        auth = self.auth_col.find()
        auth = list(auth)
        return auth

    def get_single_auth(self, key):
        # get a single auth from the auth collection
        auth = self.auth_col.find_one({"key": key})
        return auth

    def get_topics_from_collection(self, topic):
        topic = self.topics_col.find_one({"key": topic})
        if topic:
            return topic
        else:
            return {}

    def add_text_to_db(self, text_file, topic):
        with open(text_file, "r") as file:
            all_text = file.read()
            all_text = all_text.replace("-", "")
            all_text = re.sub(r"[^\S\r\n]{2,}", " ", all_text)
        while len(all_text) > 500:
            new_text_key = self.get_new_text_key()
            temp_text = all_text[:500]
            first_dot = all_text.find(".")
            dot_index = temp_text.rfind(".")
            if dot_index == -1:
                dot_index = all_text.find(".")
                if dot_index == -1:
                    dot_index = len(all_text) - 1
            temp_text = temp_text[: dot_index + 1]
            temp_text = temp_text.strip()
            key = {"key": new_text_key}
            data = {"text": temp_text, "used": False, "topic": topic}
            self.text_col.update_one(key, {"$set": data}, upsert=True)

            all_text = all_text[dot_index + 1 :]
        with open(text_file, "w") as file:
            file.write(all_text)

    def process_text(self, text, topic, doi):
        text = text.replace("-", "")
        text = re.sub(r"[^\S\r\n]{2,}", " ", text)
        while len(text) > 500:
            new_text_key = self.get_new_text_key()
            temp_text = text[:500]
            first_dot = text.find(".")
            dot_index = temp_text.rfind(".")
            if dot_index == -1:
                dot_index = text.find(".")
                if dot_index == -1:
                    dot_index = len(text) - 1
            temp_text = temp_text[: dot_index + 1]
            temp_text = temp_text.strip()
            key = {"key": new_text_key}
            data = {
                "doi": doi,
                "added_date": datetime.now(),
                "text": temp_text,
                "used": False,
                "topic": topic,
            }
            self.text_col.update_one(key, {"$set": data}, upsert=True)
            # persian_text = self.translate_with_crow(temp_text)


            text = text[dot_index + 1 :]

    def update_text_col_from_article_topic(self, topic):
        # get a random article doi
        import enchant
        from textblob import Word
        from nltk.stem import PorterStemmer, WordNetLemmatizer

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        dictionary = enchant.Dict("en_US")

        result_list = self.wrong_words_col.aggregate(
            [{"$match": {f"count_{topic}": {"$gte": 100}, "topic": topic}}]
        )
        extra_words = [result["wrong_word"] for result in result_list]
        [dictionary.add(word) for word in extra_words]

        article_dois_with_text = self.article_doi_col.find(
            {
                "topic": topic,
                "is_text": True,
                "text": {"$ne": ""},
                "text_processed": False,
            }
        )

        if article_dois_with_text:
            article_dois_with_text = list(article_dois_with_text)
            if len(article_dois_with_text) > 0:
                for article_doi in tqdm(
                    article_dois_with_text, desc="Processing", position=1, leave=False
                ):
                    cleaned_text = []
                    text = article_doi["text"]
                    text.lower()
                    text = text.replace("-", " ")
                    text = re.sub(r"\s+", " ", text)

                    words = text.split(" ")
                    for word in tqdm(words, desc="Cleaning", position=2, leave=False):
                        nltk_flag = False
                        # remove punctuation
                        processed_word = re.sub(r"[^\w\d\s]|(?![a-zA-Z]).", "", word)
                        if len(processed_word) == 0:
                            continue

                        if dictionary.check(processed_word):
                            cleaned_text.append(word)
                        else:
                            processed_word_nltk = lemmatizer.lemmatize(processed_word)
                            processed_word_nltk = stemmer.stem(processed_word_nltk)
                            nltk_flag = True
                            if dictionary.check(processed_word_nltk):
                                cleaned_text.append(word)
                            else:
                                textblob_word = Word(processed_word)
                                textblob_word_nltk = Word(processed_word_nltk)
                                result = textblob_word.spellcheck()
                                result_nltk = textblob_word_nltk.spellcheck()
                                result.extend(result_nltk)

                                for corrected_word, prob in result:
                                    if prob >= 0.85:
                                        if "." in word[-1]:
                                            corrected_word = f"{corrected_word}."
                                        elif "," in word[-1]:
                                            corrected_word = f"{corrected_word}."

                                        cleaned_text.append(corrected_word)

                            if nltk_flag:
                                self.add_word_to_wrong_word_col(
                                    processed_word, processed_word_nltk, topic
                                )
                            else:
                                self.add_word_to_wrong_word_col(
                                    processed_word, "", topic
                                )

                    text = " ".join(cleaned_text)
                    text = re.sub(r"[^a-zA-Z,.]", " ", text)
                    text = re.sub(r"\s+", " ", text)
                    text = re.sub(
                        r"\b(\w+)(?:\W\1\b)+", r"\1", text, flags=re.IGNORECASE
                    )

                    text_list = []
                    for word in text.lower().split():
                        new_word = word.strip()
                        new_word = new_word.replace(".", "")
                        new_word = new_word.replace(",", "")
                        if len(new_word) > 1 or new_word in ["a", "i"]:
                            text_list.append(word)
                    text = " ".join(text_list)

                    doi = article_doi["doi"]
                    self.process_text(text, topic, doi)
                    self.article_doi_col.update_one(
                        {"doi": doi}, {"$set": {"text_processed": True}}
                    )

    def update_text_col_from_article(self):
        max_len_topic_str = max([len(topic.value) for topic in TOPIC_ENUM], default=0)
        topic_with_text = list(self.get_topic_with_unprocessed_text())
        if len(topic_with_text) > 0:
            topic_pbar = tqdm(topic_with_text, desc="Topic", position=0, leave=True)
            for topic in topic_pbar:
                topic_pbar.set_description(f"Topic: {topic:{max_len_topic_str}}")
                self.update_text_col_from_article_topic(topic)
            print("Translate new text")
            self.batch_translate_with_crow(count=1000)

    def get_topic_with_unprocessed_text(self):
        for t in TOPIC_ENUM:
            c = self.article_doi_col.count_documents(
                {"text": {"$ne": ""}, "text_processed": False, "topic": t.value}
            )
            if c > 0:
                yield t.value

    def report_all_remaining_text(self):
        topic_dict = {}
        for topic in TOPIC_ENUM:
            topic_name = topic.value
            count = self.remaining_text(topic_name)
            topic_dict[topic_name] = count

        topic_dict = {
            k: v for k, v in sorted(topic_dict.items(), key=lambda item: item[1])
        }
        for key, value in topic_dict.items():
            print(f"{key} : {value}")



# d = Database()
