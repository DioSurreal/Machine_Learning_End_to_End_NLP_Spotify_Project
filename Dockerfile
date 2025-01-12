# Start with the official Python 3.11 image as the base image
FROM python:3.11

# Update the package list and install necessary dependencies for building packages
RUN apt-get update && apt-get install -y python3-distutils build-essential

# Upgrade pip, setuptools, and wheel to the latest versions to ensure smooth installation of Python packages
RUN pip install --upgrade pip setuptools wheel

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file from the local system to the container’s /app directory
COPY requirements.txt .

# Install the Python dependencies listed in the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK datasets for text processing
# This will download the Punkt tokenizer models
RUN python -m nltk.downloader punkt_tab

# This will download the stopwords dataset for text processing
RUN python -m nltk.downloader stopwords

# This will download the WordNet corpus for linguistic tasks
RUN python -m nltk.downloader wordnet

# Copy all files from the local directory to the container's /app directory
COPY . .

# Expose port 8080 to allow the Flask app to be accessed from outside the container
EXPOSE 8080

# Set the default command to run Gunicorn to serve the Flask app, binding it to all available IP addresses on port 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
