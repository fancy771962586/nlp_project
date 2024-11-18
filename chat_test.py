
import requests


if __name__ == '__main__':
    data={
        "messages": [
            {"role": "system", "content": "11"},
            {"role": "user", "content": "12"},
        ]
    }

    context = requests.post("http://127.0.0.1:8001/chat",json=data)
    print(context.content)