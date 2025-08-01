import glob
import google.generativeai as genai
import os
from dotenv import load_dotenv
from paddleocr import PaddleOCR
import datetime
from AIService.LangProceConversion import LanguageProcessingConversion
from DrawService.DrawImage import DrawImage
from PIL import Image
from scipy.spatial.distance import squareform, pdist

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

def process_image(image_path, orc, lp, dl):
    results = orc.predict([image_path])
    result = results[0]
    rec_texts = result['rec_texts']
    rec_scores = result['rec_scores']
    rec_polys = result['rec_polys']

    translated_results = []
    # Translate by gemini
    translated_texts = lp.translate_text(rec_texts, "vi", "Spy x Family")
    for text, translated, score, box in zip(rec_texts, translated_texts, rec_scores, rec_polys):
        print(f"EN: {text}\nVI: {translated}\nScore: {score:.2f}, Box: {box.tolist()}\n")


    # Translate by googletrans
    # translated_texts = lp.translate_texts_google(rec_texts, "vi")
    # print("translated_texts", translated_texts)
    for text, translated, score, box in zip(rec_texts, translated_texts, rec_scores, rec_polys):
        translated_results.append({
        "en": text,
        "vi": translated,
        "score": score,
        "box": box.tolist(),
    })

    time = datetime.datetime.now().strftime('%Y%m%d %H%M%S')
    output_file_name = f"{os.path.basename(image_path)}_{time}.jpg"

    dl.draw_translations_on_image(
        image_path=image_path,
        result=translated_results,
        output_path=os.path.join("./data/output", output_file_name),
        font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )

def main():
    load_dotenv()
  
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')  
    lp = LanguageProcessingConversion()
    dl = DrawImage()

    os.makedirs("./data/output", exist_ok=True)

    image_path = glob.glob("./data/img/*.[jp][png]g")
    for image_path in image_path:
        print("process image", image_path)
        try:
            process_image(image_path, ocr, lp, dl)
        except Exception as e:
            print("error", e)

if __name__ == "__main__":
    main()

