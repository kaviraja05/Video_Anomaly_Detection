import requests
try:
  resp = requests.post('http://localhost:8000/register', json={'name': 'Test', 'email': 'test@example.com', 'password': 'Password123!'})
  print(resp.status_code, resp.text)
except Exception as e: print(e)
