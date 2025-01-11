import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# โหลด NLTK data ที่จำเป็น
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

# ฟังก์ชันทำความสะอาดข้อความ
def text_cleaning(text):
    # ลบอักขระที่ไม่ต้องการ
    text = re.sub(r'Ã[\x80-\xBF]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9\s,']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # แปลงเป็นตัวพิมพ์เล็กและ tokenize
    tokens = word_tokenize(text.lower())
    return tokens

# กำหนด stopwords และฟังก์ชันลบ stopwords
custom_stopwords = {'app', 'spotify', 'listen', 'please'}

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stopwords)
    return [word for word in tokens if word not in stop_words]

# ฟังก์ชัน Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# ฟังก์ชันหลักในการ preprocess
def preprocess_text(text):
    tokens = text_cleaning(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)
