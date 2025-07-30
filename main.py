from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.cloud import vision
from google.cloud.vision_v1 import types
from paddleocr import PaddleOCR
from googletrans import Translator
import datetime
import cv2
from scipy.spatial.distance import squareform, pdist
# # Initialize the model
# llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=SecretStr(os.getenv('GEMINI_API_KEY')))
class HandleImage:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def ORC_image(self,url_image):
        image = Image.open(url_image)
        prompt1 = "get text from image and position on the image"
        response = self.model.generate_content(
            [prompt1, image],
            stream=False
        )
        print("response", response.text)
        return response.text

class LanguageProcessingConversion:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def translate_text(self, texts, target_language):
  
        numbered_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
        print("numbered_text", numbered_text)
        prompt = (
            f"Translate the following numbered English sentences into {target_language}. "
            "Return only the translations in the same numbered format.\n\n"
            f"{numbered_text}"
        )
        print("prompt", prompt)
        response = self.model.generate_content([prompt], stream=False)

      
        lines = response.text.strip().split("\n")
        translations = []
        for line in lines:
            if ". " in line:
                translations.append(line.split(". ", 1)[1])
            else:
                translations.append(line)
        return translations

    def translate_texts_google(self, texts, dest="vi"):
        translator = Translator()
        translated_texts = []

        for text in texts:
            result = translator.translate(text, dest=dest)
            translated_texts.append(result.text)

        return translated_texts

class DrawImage:
    @staticmethod
    def draw_translations_on_image(image_path, result, output_path="translated_overlay.jpg", font_path=None):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Tính các tâm của tất cả các box trước
        centers = []
        boxes = []
        for item in result:
            box = item["box"]
            boxes.append(box)
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            center_x = sum(x_coords) / 4
            center_y = sum(y_coords) / 4
            centers.append((center_x, center_y))

        # Tính ma trận khoảng cách giữa các tâm
        distances = squareform(pdist([(c[0], c[1]) for c in centers]))

        # Gom nhóm các hộp gần nhau
        threshold = 50  # Khoảng cách tối đa giữa các hộp để coi là cùng nhóm
        groups = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue
            current_group = [i]
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if not used[j] and distances[i][j] < threshold:
                    current_group.append(j)
                    used[j] = True
            if current_group:
                groups.append(current_group)

        try:
            font = ImageFont.truetype(font_path or "Arial.ttf", size=16)  # Sử dụng font hỗ trợ tiếng Việt
        except:
            font = ImageFont.load_default()

        print("result", result)
        print("groups", groups)

        # Tô trắng các nhóm và vẽ văn bản
        for idx, item in enumerate(result):
            box = item["box"]
            translated_text = item["vi"]

            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Tìm nhóm mà box hiện tại thuộc về
            group_index = next((i for i, group in enumerate(groups) if idx in group), None)
            if group_index is not None:
                group = groups[group_index]
                all_boxes = [boxes[i] for i in group]
                x_coords_group = [pt[0] for box in all_boxes for pt in box]
                y_coords_group = [pt[1] for box in all_boxes for pt in box]
                x_min_group, x_max_group = min(x_coords_group) - 10, max(x_coords_group) + 10
                y_min_group, y_max_group = min(y_coords_group) - 10, max(y_coords_group) + 10
                draw.rectangle([x_min_group, y_min_group, x_max_group, y_max_group], fill="white")

            # Vẽ văn bản tại vị trí của box hiện tại với padding
            padding = 20
            draw.rectangle(
                [x_min - padding, y_min - padding, x_max + padding, y_max + padding],
                fill="white"
            )
            draw.text((x_min, y_min), translated_text, fill="black", font=font)

        image.save(output_path)

          
        #     bbox = draw.textbbox((x_min, y_min), translated_text, font=font)
        #     text_width = bbox[2] - bbox[0]
        #     text_height = bbox[3] - bbox[1]

        #     padding = 6

            
        #     draw.rectangle(
        #         [x_min - padding, y_min - padding, x_max + padding, y_max + padding],
        #         fill="white"
        #     )

        #     draw.text((x_min, y_min), translated_text, fill="black", font=font)

        # image.save(output_path)
    @staticmethod
    def draw_translations_on_image_cv2(image_path, result, output_path="translated_overlay.jpg", font_path=None):
        image = cv2.imread(image_path)
        
        for item in result:
            box = item["box"]
            translated_text = item["vi"]

            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            padding = 20
            cv2.rectangle(
                image,
                (int(x_min - padding), int(y_min - padding)),
                (int(x_max + padding), int(y_max + padding)),
                (255, 255, 255),  
                -1 
            )

        
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            cv2.putText(
                image,
                translated_text,
                (int(x_min), int(y_min)),
                font,
                font_scale,
                (0, 0, 0), 
                thickness
            )

        cv2.imwrite(output_path, image)
       


def main():
    load_dotenv()
  
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')  
    results = ocr.predict(['./data/img/spy.png'])

    result = results[0]

    rec_texts = result['rec_texts']
    rec_scores = result['rec_scores']
    rec_polys = result['rec_polys']

    for text, score, box in zip(rec_texts, rec_scores, rec_polys):
        print(f"Text: {text}, Confidence: {score:.2f}, Box: {box.tolist()}")

    translated_results = []
    lp = LanguageProcessingConversion()
    # Translate by gemini
    # translated_texts = lp.translate_text(rec_texts, "vi")
    # for text, translated, score, box in zip(rec_texts, translated_texts, rec_scores, rec_polys):
    #     print(f"EN: {text}\nVI: {translated}\nScore: {score:.2f}, Box: {box.tolist()}\n")


    # Translate by googletrans
    translated_texts = lp.translate_texts_google(rec_texts, "vi")
    print("translated_texts", translated_texts)
    for text, translated, score, box in zip(rec_texts, translated_texts, rec_scores, rec_polys):
        translated_results.append({
        "en": text,
        "vi": translated,
        "score": score,
        "box": box.tolist(),
    })
    time = datetime.datetime.now().strftime('%Y%m%d %H%M%S')
    dl = DrawImage()

    dl.draw_translations_on_image(
        image_path="./data/img/spy.png",
        result=translated_results,
        output_path=f"translated_overlay{time}.png"
    )

    # print("result", translated_text)
    # print("response", response)



if __name__ == "__main__":
    main()

