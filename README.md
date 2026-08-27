# kip_annotation_scripts

Hand-annotation and YOLOv8n fine-tuning pipeline for the Chinese subset of **The Knowledge is Power (KIP) Dataset** — a Sino-Soviet image corpus of socialist popular-science magazines (1956–1962).

- **Dataset**: Harvard Dataverse, [doi:10.7910/DVN/KWUVPB](https://doi.org/10.7910/DVN/KWUVPB)
- **Data paper**: *The KIP Dataset: Socialist Popular Science in a Sino-Soviet Image Dataset Based on Chinese and Soviet Magazines (1956–1962)* (under review, Journal of Open Humanities Data)

## What this repository contains

The first extraction pass over the KIP source scans used a Mask R-CNN model trained on PubLayNet. Because that model is tuned to modern English-language layouts, it systematically under-segmented Chinese pages and misclassified blocks of Chinese display typography as figures. This repository documents the corrective second pass described in the data paper's Methods section:

1. **Hand-annotated training set** — 108 first-pass crops of mixed image-and-text regions drawn from the first three 1956 issues of *Zhishi jiushi liliang*, annotated with 197 bounding boxes of a single class (`image`), split train/val/test = 75/21/12 (`dataset_statistics.json` records the exact composition).
2. **YOLOv8n fine-tuning scripts** — the training pipeline that produced the deployed detector (`models/super_optimized_best_yolov8n.pt`), fine-tuned from the Ultralytics `yolov8n` checkpoint.
3. **Inference / re-extraction scripts** — the per-year extractors (`ti_*_extractor.py`) that applied the fine-tuned detector to all first-pass crops of the Chinese portion (1956–1962), re-cropping pure-image regions and recording a per-region detection confidence. These confidence values are preserved per image in the dataset's `kip_metadata.tab` (`det_conf_label`, `det_conf_score`).

## Repository structure

```
kip_annotation_scripts/
├── *.py                        # pipeline scripts (annotation, dataset processing,
│                               #   training variants, inference, per-year extractors)
├── dataset_config.yaml         # YOLO dataset config (nc=1, class: image)
├── optimized_dataset_config.yaml
├── dataset_statistics.json     # annotation-set composition (108 images / 197 boxes)
├── requirements.txt
├── annotations/                # raw per-crop annotation records (JSON)
├── processed_data/{train,val,test}/labels/   # YOLO-format label files (txt)
├── optimized_data/{train,val,test}/          # resized training images + labels
│                               #   (the complete trainable set, ~40 MB)
├── models/                     # fine-tuned weights (best_yolov8n.pt,
│                               #   super_optimized_best_yolov8n.pt) + train configs
├── results/                    # training reports (JSON)
├── runs/                       # training/validation curves and confusion matrices
├── *.log                       # training and extraction logs (provenance)
└── README_zh.md                # original Chinese project notes
```

Full-resolution training images and the original scanned pages are not distributed here; the published image corpus is available from the dataset DOI above.

## Key scripts

| Script | Role |
|---|---|
| `image_annotation_tool.py` | Bounding-box annotation tool used to build the training set |
| `dataset_processor.py` | Converts annotations into YOLO train/val/test splits |
| `train_model.py`, `optimized_train_model.py`, `final_optimized_train.py`, `super_optimized_trainer.py` | Fine-tuning variants; `super_optimized_trainer.py` produced the deployed model (YOLOv8n, 416 px, SGD, early stopping; trained on Apple MPS) |
| `inference_model.py`, `model_inference_pipeline.py` | Run the fine-tuned detector on new crops |
| `ti_{1956…1962}_*_extractor.py` | Year-by-year re-extraction of pure-image regions from first-pass crops, with confidence banding (detection floor 0.25; e.g. high ≥ 0.7, medium ≥ 0.4, low ≥ 0.25) |

## Reproducing the fine-tuning

```bash
pip install -r requirements.txt
python super_optimized_trainer.py     # trains on optimized_data/ per optimized_dataset_config.yaml
python inference_model.py             # auto-loads models/best_*.pt
```

## Rights note on training images

The 247 resized images under `optimized_data/` are cropped regions from the 1956 issues of *Zhishi jiushi liliang* (《知识就是力量》), included solely to make the annotation and fine-tuning pipeline reproducible. They are part of the same corpus published as the KIP Dataset (doi:10.7910/DVN/KWUVPB) and are subject to the terms stated on the dataset landing page; the MIT license below applies to the code in this repository, not to these images.

## Related repositories

- [`aziksh-ospanov/kip_dataset_scripts`](https://github.com/aziksh-ospanov/kip_dataset_scripts) — perceptual-hash de-duplication pipeline for the KIP corpus
- [`aziksh-ospanov/FKEA`](https://github.com/aziksh-ospanov/FKEA) — spectral clustering / reference-free evaluation toolkit used for the dataset's derived layers

## Citation

If you use this pipeline or the annotation set, please cite the KIP data paper and the dataset:

> Zhao, Ye; Ospanov, Azim; Cao, Xuenan (2026). *The KIP Dataset: Socialist Popular Science in a Sino-Soviet Image Dataset Based on Chinese and Soviet magazines (1956–1962)*. Harvard Dataverse. https://doi.org/10.7910/DVN/KWUVPB

## License

Code: MIT (see `LICENSE`). Training images: see "Rights note" above.
