from transformers import VisionEncoderDecoderModel
from transformers import ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from PIL import Image
import matplotlib.pyplot as plt

encode = 'google/vit-base-patch16-224-in21k'
decode = 'flax-community/roberta-hindi'

feature_extractor=ViTFeatureExtractor.from_pretrained(encode)
tokenizer = RobertaTokenizer.from_pretrained(decode)
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained("model")

def preview(image_path, actual_label):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    plt.imshow(image)

    file = open('output.txt', 'a')
    file.write(f"{image_path}\t{generated_text}\t{actual_label}\n")
    file.close()

    print(f"Predicted: {generated_text}, Actual: {actual_label}")
    print(generated_text)


def validate(test_file_path):
    with open(test_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            image_path = f"./dataset/{parts[0]}"
            actual_label = parts[1] if len(parts) > 1 else ''
            preview(image_path, actual_label)


test_file_path = './dataset/test.txt'

validate(test_file_path)

#image_path = "./dataset/HindiSeg/test/9/2/22.jpg"
#preview(image_path=image_path)