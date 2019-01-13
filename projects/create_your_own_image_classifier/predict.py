import json
import argparse
import numpy as np

import helper as hp

parser = argparse.ArgumentParser(description = 'predict.py')

parser.add_argument('input_img', default = 'ImageClassifier/flowers/test/5/image_05159.jpg', nargs = '*', action = "store", type = str, help = "image path")
parser.add_argument('checkpoint', default = '/home/workspace/ImageClassifier/checkpoint.pth', nargs = '*', action = "store", type = str, help = "path from where the checkpoint is loaded")
parser.add_argument('--top_k', default = 5, dest = "top_k", action = "store", type = int)
parser.add_argument('--category_names', dest = "category_names", action = "store", default = 'cat_to_name.json')
parser.add_argument('--gpu', default = "gpu", action = "store", dest = "gpu")

pa = parser.parse_args()
image_path = pa.input_img
topk = pa.top_k
gpu_cpu = pa.gpu
input_img = pa.input_img
filepath = pa.checkpoint

training_loader, testing_loader, validation_loader = hp.load_data()

# load previously saved checkpoint
hp.load_checkpoint(filepath)

# load label conversion
with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

probabilities = hp.predict(image_path, model, topk, gpu_cpu)

top5_values = np.array(probabilities[0][0])
top5_value_categories = [cat_to_name[str(i)] for i in np.array(probabilities[1][0])]

i = 0
while i < topk:
    print("{} is the category with a {} probability".format(top5_value_categories[i], top5_values[i]))
    i += 1