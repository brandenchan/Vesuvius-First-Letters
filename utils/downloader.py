import os
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from urllib.parse import urljoin

# Base URL of the directory
base_url = "http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230925090314/layers/"

# Directory to save downloaded files
save_dir = "/content/gdrive/MyDrive/modelmain/training/train scrolls/training-1/layers"
os.makedirs(save_dir, exist_ok=True)

# Authentication details
auth = ('registeredusers', 'only')

# Session to persist authentication across requests
session = requests.Session()
session.auth = auth

def download_file(file_url):
    """Function to download a single file."""
    local_filename = file_url.split('/')[-1]
    file_path = os.path.join(save_dir, local_filename)

    print(f"Downloading {local_filename}...")
    with session.get(file_url, stream=True) as file_response:
        file_response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# Get the list of files
response = session.get(base_url)
response.raise_for_status()

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')
file_urls = [urljoin(base_url, tag['href']) for tag in soup.find_all('a') if tag['href'].endswith('.tif')]

# Download files using multiple threads
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(download_file, file_urls)

print("Download completed.")
