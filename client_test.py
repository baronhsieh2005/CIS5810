# client_test.py
import requests

url = "http://127.0.0.1:8000/predict"

with open("test/ex2.png", "rb") as f:
    files = {"file": ("ex2.png", f, "image/jpeg")}
    resp = requests.post(url, files=files)

print(resp.status_code)
print(resp.text)
