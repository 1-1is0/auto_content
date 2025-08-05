import os
import time
import random
import shutil
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select

from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter
import numpy as np
import pytesseract

from database.database import Database, TOPIC_ENUM

load_dotenv()

class TextCaptcha:
    def get_captcha_text(self, filename):
        th1 = 140
        th2 = 140
        sig = 1.5

        img = Image.open(filename)
        img = img.convert("L")
        threshold = img.point(lambda p: p > th1 and 255)
        blur = np.array(threshold)
        blurred = gaussian_filter(blur, sigma=sig)
        blurred = np.array(blurred)
        blurred = Image.fromarray(blurred)
        final = blurred.point(lambda p: p > th2 and 224)
        final = final.filter(ImageFilter.EDGE_ENHANCE_MORE)
        final = final.filter(ImageFilter.SHARPEN)
        captcha_text = pytesseract.image_to_string(final)  # type: str
        captcha_text = captcha_text.strip()

        print("captcha text", captcha_text, type(captcha_text))
        captcha_input = input("Enter another captcha:")
        if captcha_input:
            captcha_text = captcha_input

        dataset_captcha_file_name = os.path.join(
            "captcha", "fix", f"captcha-{captcha_text}.png"
        )
        shutil.copy(filename, dataset_captcha_file_name)

        return captcha_text


class GenerateContent:
    URL_HOME_PAGE = os.getenv("URL_HOME_PAGE")
    URL_TRANSLATE_PAGE = os.getenv("URL_TRANSLATE_PAGE")
    URL_CONTENT_PAGE= os.getenv("URL_CONTENT_PAGE")

    def __init__(self, brand) -> None:
        self.text_captcha = TextCaptcha()
        self.db = Database()
        self.driver = webdriver.Firefox()

        self.use_translator_api = True

        auth = self.db.get_single_auth(brand)
        if auth:
            self.email = auth["email"]
            self.password = auth["password"]
        else:
            raise Exception("No Auth Found")

        self.open_login_page()

    def set_info(self, topic, allowed_topic_list=[]):
        self.topic = topic
        self.allowed_topic_list = allowed_topic_list

    def open_login_page(self):
        self.driver.get(self.URL_HOME_PAGE)
        self.original_window = self.driver.current_window_handle
        self.login_fill()
        time.sleep(20)
        if not self.use_translator_api:
            self.driver.switch_to.new_window("tab")
            self.driver.get(self.URL_TRANSLATE_PAGE)
            self.google_translate_window = self.driver.current_window_handle
            self.driver.switch_to.window(self.original_window)
        time.sleep(1)

    def generate_content(self):
        self.open_content_manage_drawer()
        time.sleep(1)
        topic_text = self.content_select_topic()
        title, first_part, last_part = self.content_title(topic_text)
        # summary
        summary_text = self.db.get_summary(first_part)
        summary_text = summary_text.format(title=title)

        text_doc = self.db.get_random_text(self.topic)
        english_text_id = text_doc["_id"]

        self.content_main_text(title, summary_text, text_doc)
        self.fill_the_detail(topic_text, title, summary_text)
        self.db.mark_text_as_done(english_text_id)
        self.db.add_one_to_monthly(self.topic)
        time.sleep(random.randint(10, 25))

    def start(self, count=20):
        # try:
        if self.db.remaining_text(self.topic) > count:
            for i in range(count):
                print(f"[{count}] Content: {i+1}")
                self.generate_content()
        else:
            print("No Text Remaining")
        # finally:
        #     self.driver.close()
        #     self.driver.quit()

    def reload_captcha_image(self):
        reload_icon = self.driver.find_element(
            By.XPATH, "/html/body/div/div[2]/div/form/div[2]/div[2]/p/a/i"
        )
        reload_icon.click()
        time.sleep(2)

    def login_fill(self):
        email_input = self.driver.find_element(By.ID, "signin-email")
        password_input = self.driver.find_element(By.ID, "signin-password")
        email_input.send_keys(self.email)
        password_input.send_keys(self.password)

        captcha_input = self.driver.find_element(By.ID, "signin-captcha")
        # captcha = int(input("Enter Page Captcha: "))

        captcha_filename = "captcha.png"
        captcha_image = self.driver.find_element(By.ID, "img_Capatcha")
        captcha_image.screenshot(captcha_filename)

        captcha_text = self.text_captcha.get_captcha_text(captcha_filename)
        # captcha_text = captcha_text.get_text_from_captcha()
        while len(captcha_text) != 5 and not captcha_text.isdigit():
            captcha_input.clear()
            self.reload_captcha_image()
            time.sleep(1)

            captcha_image = self.driver.find_element(By.ID, "img_Capatcha")
            captcha_image.screenshot(captcha_filename)
            captcha_text = self.text_captcha.get_captcha_text(captcha_filename)
            print("cond1", len(captcha_text) != 5, "cond2:", not captcha_text.isdigit())
            captcha_input.send_keys(captcha_text)

            time.sleep(2)

        captcha_input.clear()
        captcha_input.send_keys(captcha_text)

        remember_me_input = self.driver.find_element(By.ID, "accept-terms")
        remember_me_input.send_keys(Keys.RETURN)
        login_button = self.driver.find_element(By.NAME, "action_login")
        login_button.send_keys(Keys.RETURN)

    def filter_topic_with_allowed(self, topic_text):
        for allowed_topic in self.allowed_topic_list:
            if allowed_topic in topic_text:
                return True
        return False

    def content_select_topic(self):
        topics_list = self.driver.find_element(By.NAME, "user_menu_id")
        topics_list_select = Select(topics_list)
        # topics_list.send_keys(Keys.RETURN)
        # ActionChains(self.driver).click(topics_list).perform()
        topics_list.click()
        # time.sleep(2)
        topics_options = topics_list.find_elements(By.TAG_NAME, "option")
        filtered_topic = [
            topic
            for topic in topics_options
            if self.filter_topic_with_allowed(topic.text)
        ]
        selected_topic = random.choice(filtered_topic)
        selected_topic_text = selected_topic.text
        topics_list_select.select_by_visible_text(selected_topic_text)
        return selected_topic_text

    def content_title(self, topic_text):
        topic_input = self.driver.find_element(By.NAME, "title_project")
        first_part = self.db.get_random_start_title()
        last_part = self.db.get_random_end_title(self.topic)
        title = f"{first_part} {topic_text} {last_part}"
        topic_input.send_keys(title)
        return title, first_part, last_part

    def google_translate(self, text_doc) -> str:
        english_text_id = text_doc["_id"]
        english_text = text_doc["text"]
        has_persian_text = self.db.has_persian_text(english_text_id)
        if has_persian_text:
            persian_text = text_doc["persian_text"]
            return persian_text

        if self.use_translator_api:
            from pygoogletranslation import Translator

            translator = Translator()
            text = translator.translate(english_text, src="en", dest="fa")
            persian_text = text.text
        else:
            self.driver.switch_to.window(self.google_translate_window)
            english_text_area = self.driver.find_element(By.CLASS_NAME, "er8xn")
            english_text_area.clear()
            english_text_area.send_keys(english_text)
            time.sleep(10)
            translated = self.driver.find_element(By.CLASS_NAME, "lRu31")
            persian_text = translated.text
            # print(persian_text)
            self.driver.switch_to.window(self.original_window)

        self.db.save_persian_text(english_text_id, persian_text)
        return persian_text

    def content_main_text(self, title, summary_text, text_doc):
        iframe_main_text = self.driver.find_element(By.ID, "comment_ifr")
        iframe_main_text.send_keys(" ")
        iframe_main_text.send_keys(summary_text)
        iframe_main_text.send_keys(Keys.RETURN)

        text = self.google_translate(text_doc)

        iframe_main_text.send_keys(text)
        time.sleep(1)


    def open_content_manage_drawer(self):
        # content_manage_drawer = self.driver.find_element(By.ID, "ul_4")
        # content_manage_drawer.send_keys(Keys.RETURN)
        self.driver.get(self.URL_CONTENT_PAGE)
        new_content_button = self.driver.find_element(By.NAME, "btn_add_start")
        new_content_button.send_keys(Keys.RETURN)
        time.sleep(5)

    def fill_the_detail(self, topic_text, title, summary_text):
        detail_bth = self.driver.find_element(
            By.XPATH,
            "/html/body/div[2]/div[2]/div/div/div/div/div/form/div/div[12]/h3/span",
        )
        detail_bth.click()
        time.sleep(1)
        # img
        image_url = self.db.get_random_image_url(self.topic)
        img_input = self.driver.find_element(By.NAME, "images_select")
        img_input.send_keys(image_url)

        summary_input = self.driver.find_element(By.NAME, "description")
        summary_input.send_keys(summary_text)

        # tags
        # todo check for length
        tags_input = self.driver.find_element(
            By.XPATH,
            "/html/body/div[2]/div[2]/div/div/div/div/div/form/div/div[14]/div[8]/div/div/input",
        )
        tags_inputs_text_list = [
            topic_text,
            title,
        ]

        start_title = self.db.get_random_start_title()
        tags_inputs_text_list.append(f"{start_title} {topic_text}")
        start_title = self.db.get_random_start_title()
        tags_inputs_text_list.append(f"{start_title} {topic_text}")

        for text in tags_inputs_text_list:
            if len(text) < 30:
                tags_input.send_keys(text)
                tags_input.send_keys(Keys.RETURN)

        time.sleep(5)
        add_button = self.driver.find_element(By.NAME, "btn_add")
        add_button.click()


def brand_gen(brand):
    database = Database()

    auth = database.get_single_auth(brand)

    if auth:
        generate_content = GenerateContent(brand=brand)  # type: ignore
        for topic in auth["topics"]:
            print("auth topic", topic)
            topic_list = database.get_topics_from_collection(topic)
            print("topic list", topic_list)
            content_count = topic_list.get("count", 5)
            print(content_count)
            #  continue
            count = database.remaining_text(topic)
            print(topic, "remaining text", count)
            if count < content_count:
                print("not enough text for topic", topic)
                continue
            if topic_list and len(topic_list) > 0:
                allowed_topic_list = topic_list["names"]
                if allowed_topic_list and len(allowed_topic_list) > 0:
                    generate_content.set_info(
                        topic=topic, allowed_topic_list=allowed_topic_list
                    )
                    generate_content.start(content_count)

        generate_content.driver.close()
        generate_content.driver.quit()
    else:
        print("no auth found")


def main():
    brand = os.getenv("BRAND")
    try:
        brand_gen(brand)
    except Exception:
        print("Failed to generate content for", brand)

    database = Database()
    database.report_all_remaining_text()


if __name__ == "__main__":
    main()
