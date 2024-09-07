import requests


class RagFlowClient:
    def __init__(self, url):
        self.url = url

    def send_prompt(self, prompt):
        try:
            response = requests.post(f"{self.url}", data={"prompt": prompt})

            # Check if the response contains valid JSON
            if response.status_code == 200:  # Ensure the request was successful
                try:
                    return response.json()  # Attempt to parse JSON
                except ValueError:
                    return {"error": "Response was not in JSON format", "content": response.text}
            else:
                return {"error": f"Request failed with status {response.status_code}", "content": response.text}

        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

client = RagFlowClient("http://localhost:9222/chat/share?shared_id=ragflow-MxYTE1MDdhNmM3MDExZWY5OTU5ZDhjYj")
response = client.send_prompt("Hello, how are you?")
print(response)
