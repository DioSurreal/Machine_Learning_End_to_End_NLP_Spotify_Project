FROM python:3.11

# ติดตั้ง distutils และ build tools
RUN apt-get update && apt-get install -y python3-distutils build-essential

# อัปเกรด pip, setuptools และ wheel
RUN pip install --upgrade pip setuptools wheel

# ตั้งค่า working directory
WORKDIR /app

# คัดลอก requirements.txt ก่อนเพื่อใช้ cache
COPY requirements.txt .

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

# คัดลอกไฟล์ที่เหลือทั้งหมด
COPY . .

EXPOSE 8080

# รันแอปพลิเคชัน
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
