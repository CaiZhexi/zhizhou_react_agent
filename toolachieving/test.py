import http.client
import json

conn = http.client.HTTPSConnection("metaso.cn")
payload = json.dumps({"q": "红米k90", "scope": "webpage", "includeSummary": False, "size": "5", "includeRawContent": False, "conciseSnippet": False})
headers = {
  'Authorization': 'Bearer mk-xxxxxxxxxxx',
  'Accept': 'application/json',
  'Content-Type': 'application/json'
}
conn.request("POST", "/api/v1/search", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))