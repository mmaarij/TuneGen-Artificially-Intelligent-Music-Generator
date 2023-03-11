# Import the requests module for sending HTTP requests
import requests
# Import the base64 module for encoding a file to base64
import base64

def saveToGit(name, epoch, accuracy, loss):

    checkpoint_file = "./" + name
    # Set the GitHub API URL and file path
    githubAPIURL = "https://api.github.com/repos/mmaarij/TuneGen-Artificially-Intelligent-Music-Generator/contents/checkpoints/" + name
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
        with open(checkpoint_file, "rb") as f:
            encodedData = base64.b64encode(f.read())

            # Send a PUT request with the updated file content and SHA
            headers = {
                "Authorization": f'''Bearer {githubToken}''',
                "Content-type": "application/vnd.github+json"
            }
            data = {
                "message": "updated_w_api_E-" + str(epoch) + "_A-" + str(accuracy) + "_L-" + str(loss), # Put your commit message here.
                "content": encodedData.decode("utf-8"),
                "sha": sha # Include the retrieved SHA in the request
            }

            r = requests.put(githubAPIURL, headers=headers, json=data)
            print(r.text) # Printing the response
            
    elif r.status_code == 404:
        # If the file does not exist, create a new file by sending a PUT request with the required parameters
        with open(checkpoint_file, "rb") as f:
            encodedData = base64.b64encode(f.read())

            # Send a PUT request to create a new file
            headers = {
                "Authorization": f'''Bearer {githubToken}''',
                "Content-type": "application/vnd.github+json"
            }
            data = {
                "message": "created_w_api_E-" + str(epoch) + "_A-" + str(accuracy) + "_L-" + str(loss), # Put your commit message here.
                "content": encodedData.decode("utf-8")
            }

            r = requests.put(githubAPIURL, headers=headers, json=data)
            print(r.text) # Printing the response
