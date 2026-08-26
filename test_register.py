import requests


url = "http://127.0.0.1:8000/api/v1/auth/register"

payload = {
    "name": "Test User",
    "email": "test123@example.com",
    "password": "Password@123"
}

response = requests.post(url, json=payload)

print("STATUS CODE:", response.status_code)
print("RESPONSE:")
print(response.text)