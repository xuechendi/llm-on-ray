from transformers import FuyuProcessor, FuyuForCausalLM, AutoConfig, AutoModel, AutoTokenizer, TextIteratorStreamer, TextStreamer
from PIL import Image
import requests
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="my_app/test_data/bus.png")
parser.add_argument("--input", default="Generate a coco-style caption.\n")
args = parser.parse_args()

if 'WORKDIR' not in os.environ or not os.environ['WORKDIR']:
    os.environ['WORKDIR'] = os.getcwd()

# load model and processor
model_id = "adept/fuyu-8b"
if os.path.exists(os.path.join(os.environ['WORKDIR'], "models", model_id)):
    model_id = os.path.join(os.environ['WORKDIR'], "models", model_id)

processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id)

# prepare inputs for the model

# url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
# image = Image.open(requests.get(url, stream=True).raw)

text_prompt = args.input
img_path = args.path

url = os.path.join(os.environ['WORKDIR'], img_path)
image = Image.open(url)
inputs = processor(text=text_prompt, images=image, return_tensors="pt")

# autoregressively generate text
streamer = TextStreamer(processor, skip_prompt=True, skip_special_tokens=True)
_ = model.generate(**inputs, streamer = streamer, max_new_tokens=128)

#generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
#print(generation_text)