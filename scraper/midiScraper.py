import requests
from bs4 import BeautifulSoup
import os

genre = "pop"

# Make a request to the website
url = "https://freemidi.org/genre-" + genre
response = requests.get(url)

# Parse the HTML content of the website using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all divs on the page with class "genre-link-text"
link_divs = soup.find_all("div", class_="genre-link-text")

# Extract the href attribute from each div
links = [div.a["href"] for div in link_divs]

# Print the list of links
#print(links)

base_url = "https://freemidi.org/"

# Make a request to the website and extract the download links
def extract_download_links(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    spans = soup.find_all("span", itemprop="name")
    return [span.a["href"] for span in spans]

# Visit each link and extract the download links
download_links = []
for link in links:
    link = base_url+link
    download_link_arr = extract_download_links(link)
    print(download_link_arr)
    download_links.extend(download_link_arr)

# Print the list of download links
#print(download_links)


downloadmidi_links = []

for link in download_links:
    full_link = base_url + link
    response = requests.get(full_link)
    soup = BeautifulSoup(response.content, 'html.parser')
    downloadmidi_element = soup.find('a', {'id': 'downloadmidi'})
    if downloadmidi_element is not None:
        downloadmidi_href = downloadmidi_element.get('href')
        downloadmidi_links.append(downloadmidi_href)

print(downloadmidi_links)

# create the genre directory if it doesn't already exist
if not os.path.exists(genre):
    os.mkdir(genre)


headers = {
            "Host": "freemidi.org",
            "Connection": "keep-alive",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
           }


print("****************************************")
print("****************************************")
print("****************************************")
print("****************************************")
print("Number of Files To Download: ", len(downloadmidi_links))
print("****************************************")
print("****************************************")
print("****************************************")
print("****************************************")


# Download each MIDI file and save it into the genre directory
for i, link in enumerate(downloadmidi_links):
    filename = f"{genre}_{i}.mid"
    full_link = base_url + link

    session = requests.Session()

    #the website sets the cookies first
    req1 = session.get(full_link, headers = headers)

    #Request again to download
    req2 = session.get(full_link, headers = headers)

    with open(f"./{genre}/{filename}", "wb") as saveMidi:
        saveMidi.write(req2.content)