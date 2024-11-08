import subprocess
import json
import tempfile
from PIL import ImageGrab
import base64

class OLlamaModel:
    def __init__(self, model_name="llava-phi3"):
        self.model_name = model_name

    def encode_image(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def call_model(self, user_prompt, system_prompt=None, image_paths=None):
        if image_paths is not None:
            encoded_images = list(map(self.encode_image, image_paths))
            user_prompt = {"role": "user", "content": user_prompt, "images": encoded_images}
        else:
            user_prompt = {"role": "user", "content": user_prompt}
        
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append(user_prompt)

        json_data = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "stream": False
        })

        result = subprocess.run(
            ["curl", "http://localhost:11434/api/chat", "-d", "@-"],
            input=json_data,
            capture_output=True,
            text=True
        )
        
        json_output = result.stdout.strip()
        data = json.loads(json_output)
        response_string = data.get("message", {}).get("content")
            


        response_string = data.get("message", {}).get("content")
        assert response_string is not None, "Make sure OLlama is turned on!"
        return response_string

    def send_screenshot_to_model(self, user_prompt, system_prompt=None):
        
        screenshot_path = "screenshot.png"
        screenshot = ImageGrab.grab()
        screenshot.save(screenshot_path, "PNG")

        response = self.call_model(user_prompt, system_prompt=system_prompt, image_paths=[screenshot_path])
        
        return response

if __name__ == "__main__":
    ollama_model = OLlamaModel(model_name="llava-phi3")
    response = ollama_model.send_screenshot_to_model("Can you help analyze my screen?")
    print(response)