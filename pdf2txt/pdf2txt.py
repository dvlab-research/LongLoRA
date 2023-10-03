# Written by Shaozuo Yu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import pdf2image
import easyocr
import cv2
from config import add_vit_config
from backbone import build_vit_fpn_backbone
import torch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.layers import nms
import pickle
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
import time

from symspellpy.symspellpy import SymSpell
import pkg_resources
import re


prefix_length = 7
sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=prefix_length)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
# filter
regex = "[A-Za-z0-9=:/\*]*[=:+-][A-Za-z0-9=:/\*]"


def detect_objects(image_path, predictor, cfg):
    # Step 5: run inference
    img = cv2.imread(image_path)

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    md.set(thing_classes=["text", "title", "list", "table", "figure"])

    start_time = time.time()

    detections = predictor(img)["instances"]

    end_time = time.time()

    print(f"detection model部分执行时间: {end_time - start_time} 秒")
    # get boxes and scores
    boxes = detections.pred_boxes.tensor
    scores = detections.scores

    # NMS
    keep = nms(boxes, scores, 0.1)
    detections = detections[keep]
    scores = detections.scores

    threshold = 0.8  # you can adjust this value
    keep2 = torch.nonzero(scores > threshold).squeeze(1)
    detections = detections[keep2]

    return detections

def process_pdf(pdf_file, outputs_dir, config_file):

    results = {}

    tmp_dir = os.path.join(outputs_dir, 'tmp')
    txt_dir = os.path.join(outputs_dir, 'txt')
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    #load detection model
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    reader = easyocr.Reader(['en'], gpu=True)

    book_name = os.path.splitext(pdf_file)[0]
    book_base_name = os.path.basename(pdf_file)
    txt_file_path = os.path.join(txt_dir, f"{book_base_name}.txt")
    if os.path.exists(txt_file_path):
        raise ValueError(f"Skipping {book_name} as it already exists in the output directory.")

    start_time = time.time()

    book_name = os.path.splitext(pdf_file)[0]
    images = pdf2image.convert_from_path(pdf_file)

    end_time = time.time()

    print(f"pdf2image time: {end_time - start_time} s")

    book_results = []
    for page_num, image in tqdm(enumerate(images, start=1), desc=f"Processing {book_name}", leave=False):
        image_path = os.path.join(tmp_dir, f"{book_base_name}-{page_num}.png")
        image.save(image_path)

        start_time = time.time()

        detections = detect_objects(image_path, predictor, cfg)

        end_time = time.time()

        print(f"detection time: {end_time - start_time} s")

        boxes = detections.pred_boxes.tensor.tolist()
        labels = detections.pred_classes.tolist()

        # get boxes
        all_detections = [(bbox, label_id) for bbox, label_id in zip(boxes, labels)]

        # sort
        all_detections.sort(key=lambda x: (x[0][1], x[0][0]))

        start_time = time.time()

        label_counter = {"figure": 0, "table": 0, 'text': 0, 'list': 0, 'title': 0}
        for bbox, label_id in all_detections:
            #print("Number of classes:", len(MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes))
            label = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes[label_id]
            cropped_image_np = np.array(image.crop(bbox))

            if label in ['text', 'list', 'title']:
                #reader = easyocr.Reader(['en'], cudnn_benchmark=True)
                ocr_result = reader.readtext(cropped_image_np, batch_size=10)
                extracted_text = ' '.join([item[1] for item in ocr_result])

                # SymSpell for word segmentation
                suggestions = sym_spell.word_segmentation(extracted_text)
                segmented_text = suggestions.corrected_string

                # filter
                filtered_text = re.sub(regex, "", segmented_text)

                book_results.append(extracted_text)

        end_time = time.time()

        print(f"ocr time: {end_time - start_time} s")

    results[book_name] = book_results
    with open(os.path.join(txt_dir, f"{book_base_name}.txt"), 'w') as f:
        f.write('\n'.join(book_results))

    # delete tmp dir
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PDF processing script")
    parser.add_argument("--pdf_path", help="Path to PDF file", type=str, required=True)
    parser.add_argument("--outputs_dir", help="Directory to save outputs", type=str, required=True)
    parser.add_argument("--config_file", default="configs/cascade_dit_large.yaml", metavar="FILE", help="path to config file")

    args = parser.parse_args()
    
    process_pdf(args.pdf_path, args.outputs_dir, args.config_file)
