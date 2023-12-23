import os
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from urllib.parse import urljoin


BASE_URL = "http://dl.ash2txt.org/"
SCROLL_SUFFIX = "full-scrolls/Scroll1.volpkg/paths/"

# Authentication details
AUTHENTICATION = ('registeredusers', 'only')

# Session to persist authentication across requests
session = requests.Session()
session.auth = AUTHENTICATION

def download_segment_files(
    segment_id: str,
    save_dir: str,
    base_url: str = BASE_URL,
    scroll_suffix: str = SCROLL_SUFFIX,
    max_workers=4
):
    """Given a segment id, this function will download all the layers and the mask."""

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    segment_dir = os.path.join(save_dir, segment_id)
    if not os.path.exists(segment_dir):
        os.mkdir(segment_dir)

    layers_dir = os.path.join(segment_dir, "layers")
    if not os.path.exists(layers_dir):
        os.mkdir(layers_dir)

    # Get urls of files to download
    print(f"Downloading layers and mask for {segment_id}.")
    segment_folder_url = os.path.join(base_url, scroll_suffix, segment_id)
    segment_layers_folder_url = os.path.join(segment_folder_url, "layers")

    mask_url = list_urls(
        url=segment_folder_url,
        suffix="_mask.png"
    )
    layers_urls = list_urls(
        url=segment_layers_folder_url,
        suffix=".tif"
    )

    file_urls = mask_url + layers_urls
    num_files = len(file_urls)
    save_paths = [segment_dir] + [layers_dir] * len(layers_urls)


    # Download files using multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(
            download_file,
            file_urls,
            save_paths,
            [segment_id] * num_files
        )
    print("Download completed.")


def download_file(file_url, save_dir, segment_id):
    """Function to download a single file."""
    local_filename = file_url.split('/')[-1]
    file_path = os.path.join(save_dir, local_filename)

    with session.get(file_url, stream=True) as file_response:
        file_response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {local_filename}")


def list_urls(
    url,
    suffix
):
    """Get the list of files in a http://dl.ash2txt.org/ url"""

    # Append slash to end of url if there isn't one so that the call to urljoin below
    # works as it should
    if url[-1] != "/":
        url = url + "/"

    response = session.get(url)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    file_urls = []
    a_tags = soup.find_all("a")
    for tag in a_tags:
        if tag['href'].endswith(suffix):
            file_urls.append(urljoin(url, tag['href']))
    return file_urls
