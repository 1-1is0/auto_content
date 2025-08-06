import json
import ebooklib
from ebooklib import epub
import re
from collections import Counter
from bs4 import BeautifulSoup
import pandas as pd
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


class COLLECTION_ENUM(Enum):
    WORDS_DB = "words_db"
    WORDS_COL = "words_col"


class WORDS_COL_ENUM(Enum):
    WORD = "word"
    TRANSLATION = "translation"  # a json
    TRANSLATOR = "translator"  # source of translation
    KNOWN = "known"  # is known to the user
    SOURCES = "sources"  # a list
    COUNT = "count"


class Database:
    def __init__(self) -> None:
        self.mongo_client = self.connect_to_db()
        self.init_database()
        # print("Remaining Text Documents are:", self.remaining_text())

    def connect_to_db(self) -> pymongo.MongoClient:
        username = urllib.parse.quote_plus("admin")
        password = urllib.parse.quote_plus("7qQtr)m7V;6REZyr")
        host = "localhost"
        port = "27017"
        mongo_client: pymongo.MongoClient = pymongo.MongoClient(
            f"mongodb://{username}:{password}@{host}:{port}"
        )
        return mongo_client

    def init_database(self):
        self.words_db = self.mongo_client[COLLECTION_ENUM.WORDS_DB.value]
        self.words_col = self.words_db[COLLECTION_ENUM.WORDS_COL.value]

    def add_word_to_words_col(self, word, source, count):
        key = {WORDS_COL_ENUM.WORD.value: word}
        word_doc = self.words_col.find_one(key)
        if not word_doc:
            data = {
                **key,
                WORDS_COL_ENUM.TRANSLATION.value: None,
                WORDS_COL_ENUM.KNOWN.value: False,
                WORDS_COL_ENUM.TRANSLATOR.value: None,
                WORDS_COL_ENUM.SOURCES.value: [source],
                WORDS_COL_ENUM.COUNT.value: count,
            }
            self.words_col.insert_one(data)
        else:
            sources_list = word_doc[WORDS_COL_ENUM.SOURCES.value]
            # just add the new source and update the count
            data = {"$addToSet": {WORDS_COL_ENUM.SOURCES.value: source}}
            if source not in sources_list:
                data["$inc"] = {WORDS_COL_ENUM.COUNT.value: count}
            self.words_col.update_one(key, data)

    def translate_with_crow(self, word):
        command = [
            "flatpak",
            "run",
            "io.crow_translate.CrowTranslate",
            "-s",
            "en",
            "-t",
            "fa",
            "--json",
            word,
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE)
        translate = result.stdout
        translate = translate.decode("unicode-escape").encode("latin1").decode("utf-8")
        translate = str(translate)

        if len(translate) > 1:
            translate = json.loads(translate)
            return translate

        raise Exception("Can't translate")

    def mark_as_known(self, word_doc):
        pass

    def mark_as_unknow(self, word_doc):
        data = {"translator": "crow"}
        pass

    def iterate_to_mark_known(self):
        pass

    def add_list_of_words(self, words_dict, source):
        for word, count in words_dict.items():

            self.add_word_to_words_col(word=word, count=count, source=source)
    
    def get_list_of_words(self, limit):
        res = self.words_col.find({"count": {"$gte": limit}})
        return res
        


def chapter_to_str(chapter):
    soup = BeautifulSoup(chapter.get_body_content(), "html.parser")
    # to return only the p tags
    # text = [para.get_text() for para in soup.find_all("p")]
    text = soup.text.replace("\n", " ").strip()
    return text


def lowercase_capitalized_words(text):
    def convert_word(match):
        word = match.group(0)
        if word.isupper():
            return word.lower()
        return word

    pattern = r"\b[A-Z]+\b"
    result = re.sub(pattern, convert_word, text)
    return result


def lowercase_first_word(text):
    # for the first word in a sentect (ex I go to gym. I get tierd.)
    def convert_first_word(match):
        word = match.group(0)
        return word[0].lower() + word[1:]

    pattern = r"(?:^|[.!?]\s*)(\w+)"
    result = re.sub(pattern, convert_first_word, text)
    return result


import nltk
from nltk.corpus import wordnet


def lowercase_except_names(input_string):
    words = nltk.word_tokenize(input_string)
    tagged_words = nltk.pos_tag(words)
    named_entities = nltk.ne_chunk(tagged_words)

    result_words = []
    for subtree in named_entities.subtrees():
        if subtree.label() == "PERSON":
            result_words.extend([word for word, tag in subtree.leaves()])
        else:
            result_words.extend([word.lower() for word, tag in subtree.leaves()])

    return " ".join(result_words)


def singularize_plurals(input_string):
    words = nltk.word_tokenize(input_string)

    result_words = []
    for word in words:
        lemma = wordnet.morphy(word)
        if lemma is None:
            result_words.append(word)  # Word not found in WordNet, add as is
        else:
            result_words.append(lemma)

    return " ".join(result_words)


def process_text(text):
    text = re.sub(r"[^\S\r\n]{2,}", " ", text)
    text = lowercase_except_names(text)

    remove_character = ".,()@#%!?*^&‘’\\/:"
    tmp = ""
    for c in remove_character:
        tmp += f"\\{c}"
    remove_character = tmp

    # remove special characters
    text = re.sub(rf"[{remove_character}]", " ", text)
    # remove any digits
    text = re.sub(r"\d+", " ", text)
    # remove extra space and new lines (more that one)
    text = re.sub(r"[^\S\r\n]{2,}", " ", text)

    # only lower all capitilize words (not Jim)
    # text = lowercase_capitalized_words(text)
    text = singularize_plurals(text)
    text = text.split(" ")
    # remove small words
    text = [t for t in text if len(t) > 2]

    counter = Counter(text)
    print(counter.most_common(10))
    print(len(set(text)))
    # for k, v in counter.items():
    #     print(k, v)
    df = pd.DataFrame(list(counter.items()))
    df.to_csv("vocabs_output.csv")
    return counter


def read_epub_books(file_path):
    database = Database()
    book_name = file_path.split("/")[-1]
    book = epub.read_epub(file_path)
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    text = " "
    for chapter in items:
        text += chapter_to_str(chapter)
    print(len(text))
    words_dict = process_text(text)
    database.add_list_of_words(words_dict=words_dict, source=book_name)
