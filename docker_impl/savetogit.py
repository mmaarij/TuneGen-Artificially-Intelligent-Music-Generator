# Import the requests module for sending HTTP requests
import requests
# Import the base64 module for encoding a file to base64
import base64

def saveToGit():
    # Set the GitHub API URL and file path
    githubAPIURL = "https://api.github.com/repos/mmaarij/TuneGen-Artificially-Intelligent-Music-Generator/contents/checkpoints/model_checkpoint.h5"
    # Replace "bracketcounters" with your username, replace "test-repo" with your repository name and replace "new-image.png" with the filename you want to upload from local to GitHub.

    # Paste your API token here
    githubToken = "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"

    # Check if the file exists by sending a GET request
    headers = {
        "Authorization": f'''Bearer {githubToken}''',
        "Content-type": "application/vnd.github+json"
    }
    r = requests.get(githubAPIURL, headers=headers)
    if r.status_code == 200:
        # If the file exists, retrieve its SHA and send a PUT request to update it
        response = r.json()
        content = response["content"]
        sha = response["sha"]

        # Open the local file to be uploaded
        with open("./model_checkpoint.h5", "rb") as f:
            encodedData = base64.b64encode(f.read())

            # Send a PUT request with the updated file content and SHA
            headers = {
                "Authorization": f'''Bearer {githubToken}''',
                "Content-type": "application/vnd.github+json"
            }
            data = {
                "message": "checkpoint_updated_w_api", # Put your commit message here.
                "content": encodedData.decode("utf-8"),
                "sha": sha # Include the retrieved SHA in the request
            }

            r = requests.put(githubAPIURL, headers=headers, json=data)
            print(r.text) # Printing the response
            
    elif r.status_code == 404:
        # If the file does not exist, create a new file by sending a PUT request with the required parameters
        with open("./model_checkpoint.h5", "rb") as f:
            encodedData = base64.b64encode(f.read())

            # Send a PUT request to create a new file
            headers = {
                "Authorization": f'''Bearer {githubToken}''',
                "Content-type": "application/vnd.github+json"
            }
            data = {
                "message": "checkpoint_created_w_api", # Put your commit message here.
                "content": encodedData.decode("utf-8"),
                "path": "checkpoints/model_checkpoint.h5"
            }

            r = requests.put(githubAPIURL, headers=headers, json=data)
            print(r.text) # Printing the response