import requests
import json

data = "public static String unEscapeString(String str,char escapeChar,char charToEscape){\n  return unEscapeString(str,escapeChar,new char[]{charToEscape});\n}\n"

headers = {
    'content-type': 'application/json',
    'Connection': 'keep-alive'
}
r = requests.get("http://127.0.0.1:9000/test")
# r = requests.post("https://127.0.0.1:9000/test", data=json.dumps(data), headers=headers)

print(r.json())