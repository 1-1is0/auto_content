import matplotlib.pyplot as plt
import cv2
import shutil
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter
import numpy as np
from PIL import Image
import pytesseract
import random
from glob import glob
import os


def get_an_image():
    captcha_dir = "captcha"
    filename_list = glob(os.path.join("..", captcha_dir, "img-*.png"))
    filename_list = sorted(filename_list, key=lambda x: int(
        x.split("-")[-1].split(".")[0]), reverse=True)
    print(len(filename_list), "Images left")
    a_file_name = filename_list[0]
    return a_file_name


def move_file(file_name, image_code):
    dest = os.path.join("..", "captcha", "fix")

    os.makedirs(dest, exist_ok=True)
    new_file_name = os.path.join(dest, f"captcha-{image_code}.png")
    # check if file exists and change the destination file name
    if os.path.exists(new_file_name):
        files = glob(os.path.join(dest, f"captcha-{image_code}*.png"))
        last_file_number = len(files)
        new_file_name = os.path.join(
            dest, f"captcha-{image_code}-{last_file_number}.png")
        print("Duplicated file, New file name", new_file_name)

    # move a file from a to b
    shutil.move(file_name, new_file_name)


def get_captcha_text(filename):

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
    captcha_text = pytesseract.image_to_string(
        final, config='-c  tessedit_char_whitelist=0123456789 --psm 6')  # type: str
    return captcha_text.strip()


def get_captcha_text2(filename):
    original_img = cv2.imread(filename)  # Load the upsampled image
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(img, np.array([50, 50, 50]), np.array([179, 255, 255]))
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dlt = cv2.dilate(msk, krn, iterations=1)
    thr = 255 - cv2.bitwise_and(dlt, msk)

    txt = pytesseract.image_to_string(
        thr, config='-c  tessedit_char_whitelist=0123456789 --psm 6')
    txt = txt.strip()
    if len(txt) == 5:
        return txt
    return None


def main():
    for i in range(100):
        filename = get_an_image()
        print(filename)
        img = Image.open(filename)

        solution = get_captcha_text(filename)
        solution2 = get_captcha_text2(filename)

        plt.imshow(img)
        plt.show(block=False)
        print("1: ", solution)
        print("2: ", solution2)
        confirm = input(f"[1/2/n] default n: ").lower()
        if confirm == "1" and solution and solution.isdigit():
            image_code = solution
        elif confirm == "2" and solution2 and solution2.isdigit():
            image_code = solution2
        elif len(confirm) == 5 and confirm.isdigit():
            image_code = confirm
        else:
            image_code = input("Enter the code: ")
        int(image_code)
        move_file(filename, image_code)
        plt.clf()
    plt.close()


def scan_all_images():
    captcha_dir = "captcha"
    filename_list = glob(os.path.join("..", captcha_dir, "img-*.png"))
    filename_list = sorted(filename_list, key=lambda x: int(
        x.split("-")[-1].split(".")[0]), reverse=True)
    count = 0
    for i, filename in enumerate(filename_list):
        print(f"[{i}/{len(filename_list)}]")
        solution = get_captcha_text(filename)
        solution2 = get_captcha_text2(filename)
        if len(solution) == 5 and solution.isdigit():
            img = Image.open(filename)
            plt.imshow(img)
            plt.show(block=False)
            print("1: ", solution)
            print("2: ", solution2)
            confirm = input(f"[1/2/n] default n: ").lower()
            if confirm == "1":
                image_code = solution
            elif confirm == "2":
                image_code = solution2
            else:
                image_code = input("Enter the code: ")
            count += 1
            move_file(filename, image_code)
            plt.clf()

    print(count)


if __name__ == "__main__":
    # scan_all_images()
    main()
