# Extract text from pdf by dit detection and ocr

The script uses various libraries such as `pdf2image`, `easyocr`, `ditod` and `detectron2` for processing.
Detected objects are categorized into "text", "title", "list", "table", and "figure".
The script provides detailed timing information for various processing steps, which can be useful for performance analysis.
Text extraction uses `easyocr` and the results are further processed using SymSpell for word segmentation and a regular expression for filtering.

1. install Requirements:
```
python -m pip install -r requirements.txt
```
Note detectron2 may need to install by python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

2. An example for basic usage:
```
python pdf2txt.py --pdf_path /path/to/txt  --outputs_dir /path/to/output_dir
```

Please download the weight [trained_ocr_cascade_large.pth](https://drive.google.com/file/d/1DtHtR3hhj8Df_Lkgdm9P79Eljot5MR_i/view?usp=share_link) first.

