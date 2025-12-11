# client_test.py
import requests

url = "http://127.0.0.1:8000/elbow_flare"

with open("images/nf/nf5.png", "rb") as f:
    files = {"file": ("images/nf/nf5.png", f, "image/jpeg")}
    resp = requests.post(url, files=files)

print(resp.status_code)
print(resp.text)
