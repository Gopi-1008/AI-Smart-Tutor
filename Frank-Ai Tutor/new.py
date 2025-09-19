from PIL import Image
import pytesseract

# Optional if you installed "just for me" and PATH isn't working
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\<YourName>\AppData\Local\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
img = Image.open("bite2.png")  # Replace with your test image path
text = pytesseract.image_to_string(img)
print(text)
