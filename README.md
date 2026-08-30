# kip_annotation_scripts

Hand-annotation and YOLOv8n fine-tuning pipeline for the Chinese subset of **The Knowledge is Power (KIP) Dataset** — a Sino-Soviet image corpus of socialist popular-science magazines (1956–1962).

- **Dataset**: Harvard Dataverse, [doi:10.7910/DVN/KWUVPB](https://doi.org/10.7910/DVN/KWUVPB)
- **Data paper**: *The KIP Dataset: Socialist Popular Science in a Sino-Soviet Image Dataset Based on Chinese and Soviet Magazines (1956–1962)* (under review, Journal of Open Humanities Data)

## What this repository contains

The first extraction pass over the KIP source scans used a Mask R-CNN model trained on PubLayNet. Because that model is tuned to modern English-language layouts, it systematically under-segmented Chinese pages and misclassified blocks of Chinese display typography as figures. This repository documents the corrective second pass described in the data paper's Methods section:

1. **Hand-annotated training set** — 108 first-pass crops of mixed image-and-text regions drawn from the first three 1956 issues of *Zhishi jiushi liliang*, annotated with 197 bounding boxes of a single class (`image`); `dataset_statistics.json` records the composition.
2. **YOLOv8n fine-tuning scripts** — the training pipeline that produced the deployed detector (`models/super_optimized_best_yolov8n.pt`), fine-tuned from the Ultralytics `yolov8n` checkpoint. See the data-split note below: the deployed detector was in effect trained on all 108 annotated images, and no held-out evaluation is reported.
3. **Inference / re-extraction scripts** — the per-year extractors that applied the fine-tuned detector to all first-pass crops of the Chinese portion (1956–1962), re-cropping pure-image regions and recording a per-region detection confidence. These confidence values are preserved per image in the dataset's `kip_metadata.tab` (`det_conf_label`, `det_conf_score`). See "Per-year extraction scripts" below for the exact script and banding thresholds used for each year.

## Data-split note (no held-out evaluation)

An initial 75/21/12 train/val/test split of the 108 annotated images was produced by `dataset_processor.py`. However, the splitter was originally run several times without a fixed random seed and without clearing previously written split directories, so images accumulated across splits: in the shipped `processed_data/` and `optimized_data/` directories, every val and test image is also present in train. The deployed detector (`models/super_optimized_best_yolov8n.pt`) was therefore in effect fine-tuned on the complete annotation set, and any validation curves produced against these contaminated splits are not generalization estimates (they have been removed from this repository). The retained `*.log` files still quote metric values from those sessions (e.g. `mAP50 ≈ 0.96` in `super_optimized_training.log` and `inference.log`, computed against the contaminated 82/57-image val/test splits); they are kept verbatim as session records and **must not be cited as detector accuracy**.

The contamination also left three stale label files behind: `1956_Issue-1_1_page_0_fig0_fig1` and `1956_Issue-1_21_page_0_fig0_fig0` in `val/labels/`, and `1956_Issue-1_1_page_0_fig0_fig0` in `test/labels/`, carry earlier annotation versions that differ from the current `annotations/` records (and from their own train copies) — the annotations were revised between splitter runs. For any reuse, treat `annotations/` as the authoritative version of all 197 boxes.

We keep the contaminated split directories as shipped because they are the exact training input of the deployed detector. Detector quality was not established by held-out metrics but by the downstream procedure described in the data paper's Methods: every re-extracted crop of the Chinese portion was manually re-cleaned. `dataset_processor.py` in this repository has since been fixed (seeded shuffle, split directories cleared before writing), so re-running it on a machine that has the original `raw_images/` crops (not distributed here) produces a clean, disjoint 75/21/12 split — note that a model retrained on such a split will differ from the deployed detector. **Warning:** re-running `dataset_processor.py` overwrites the shipped `processed_data/` split directories and rewrites `dataset_statistics.json`, i.e. it destroys the contamination evidence documented above; run it on a separate copy if you want to preserve the shipped state.

## Annotation quality note

16 of the 197 bounding boxes (8.1%, all in the 1956 files) extend slightly past the image border — negative coordinates of −2 to −3 px, or `x2`/`y2` exceeding the image size by up to 19 px (worst case: `annotations/1956_Issue-2_19_page_0_fig0_fig1.json`, `x2 = 1420` against width 1401). The cause was a missing clamp in `image_annotation_tool.py`'s canvas-to-image coordinate conversion (since fixed in this repository); the out-of-bounds values propagated verbatim into the shipped YOLO label files, where 44 label lines have normalized coordinates slightly outside `[0, 1]`. The overshoot is at most ~1.4% of the image dimension and Ultralytics clips such boxes internally during training, but **clip the boxes to the image bounds before reusing the annotations in other pipelines**. The shipped annotation and label files are kept as-is because they are the exact training input of the deployed detector (see the data-split note above).

## Repository structure

```
kip_annotation_scripts/
├── *.py                        # pipeline scripts (annotation, dataset processing,
│                               #   training variants, inference, per-year extractors)
├── dataset_config.yaml         # YOLO dataset config (nc=1, class: image)
├── optimized_dataset_config.yaml
├── dataset_statistics.json     # annotation-set composition (108 images / 197 boxes;
│                               #   see data-split note)
├── requirements.txt
├── annotations/                # raw per-crop annotation records (JSON)
├── processed_data/{train,val,test}/labels/   # YOLO-format label files (txt)
├── optimized_data/{train,val,test}/          # resized training images + labels
│                               #   (the complete trainable set, ~58 MB)
├── models/                     # fine-tuned weights (best_yolov8n.pt,
│                               #   super_optimized_best_yolov8n.pt) + train configs
├── results/                    # training reports (kept verbatim as provenance — none
│                               #   contains evaluation metrics; dataset_info blocks,
│                               #   where present, describe the intended 75/21/12 split,
│                               #   not the contaminated on-disk one; the super_optimized
│                               #   report was truncated by a force-stop and is not
│                               #   valid JSON)
├── *.log                       # training and extraction logs (provenance)
└── README_zh.md                # original Chinese project notes
```

Full-resolution training images and the original scanned pages are not distributed here; the published image corpus is available from the dataset DOI above.

## Key scripts

| Script | Role |
|---|---|
| `image_annotation_tool.py` | Bounding-box annotation tool used to build the training set |
| `dataset_processor.py` | Converts annotations into YOLO train/val/test splits (now seeded and self-cleaning; see data-split note) |
| `train_model.py`, `optimized_train_model.py`, `final_optimized_train.py`, `super_optimized_trainer.py` | Fine-tuning variants; `super_optimized_trainer.py` produced the deployed model (YOLOv8n, 416 px, SGD, early stopping; trained on Apple MPS) |
| `inference_model.py`, `model_inference_pipeline.py` | Run the fine-tuned detector on new crops; `inference_model.py` loads the deployed `models/super_optimized_best_yolov8n.pt` when present |
| per-year extractors (table below) | Year-by-year re-extraction of pure-image regions from first-pass crops |

## Per-year extraction scripts

Each year of the Chinese portion was processed from two working pools of first-pass crops (working folders `{year} TI` and `{year} Multi` on the processing machine). All extractors load the same deployed detector (`models/super_optimized_best_yolov8n.pt`) with a uniform detection floor of `conf = 0.25`, but the **confidence-band labels were tuned per script and are not comparable across years or pools** — for cross-year comparisons use the raw `det_conf_score` in `kip_metadata.tab`, not `det_conf_label` (the dataset README makes the same point: banding thresholds are heuristic and version-specific).

| Year | TI-pool script (band floors) | Multi-pool script (band floors) |
|---|---|---|
| 1956 | `ti_1956_original_extractor.py` (premium .90 / excellent .80 / high .65 / good .45 / low .25) | `ti_1956_multi_extractor.py` (same bands as TI) |
| 1957 | `ti_1957_ultimate_extractor.py` (ultra_premium .98 / premium .90 / excellent .80 / high .65 / good .50 / medium .35 / low .25) | `ti_1957_multi_extractor.py` (premium .90 / excellent .80 / high .65 / good .45 / low .25) |
| 1958 | `ti_1958_ultimate_extractor.py` (premium .95 / excellent .85 / high .70 / good .50 / medium .30 / low .25) | `ti_1958_multi_extractor.py` (premium .90 / excellent .80 / high .65 / good .45 / low .25) |
| 1959 | `ti_1959_comprehensive_extractor.py` (excellent .90 / high .70 / good .50 / medium .30 / low .25) | `ti_1959_multi_extractor.py` (same bands as TI) |
| 1960 | `ti_1960_ultimate_extractor.py` (premium .95 / excellent .85 / high .70 / good .50 / medium .30 / low .25) | `ti_1960_multi_extractor.py` (premium .90 / excellent .80 / high .60 / good .40 / low .25) |
| 1961 | `real_world_image_extractor.py` (no banding; raw confidence only, floor .25) | `multi_dataset_extractor.py` (high .70 / medium .40 / low .25) |
| 1962 | `ti_1962_dataset_extractor.py` (high .80 / good .60 / medium .40 / low .25) | `ti_1962_multi_extractor.py` (high .70 / medium .40 / low .25) |

The extractors read the original first-pass crops from machine-specific working folders that are not distributed here (the published corpus is at the dataset DOI above); they are kept verbatim, together with their `*.log` files, as provenance of how each year was processed.

## Reproducing the fine-tuning

```bash
pip install -r requirements.txt
python super_optimized_trainer.py     # trains on optimized_data/ per optimized_dataset_config.yaml
python inference_model.py             # loads models/super_optimized_best_yolov8n.pt;
                                      #   edit input_folder in main() to point at your own crops
```

`super_optimized_trainer.py` (the script that produced the deployed model) resolves its paths relative to the repository root and trains directly on the shipped `optimized_data/`; its preprocessing step reports 0 processed images because `processed_data/` ships labels only — this is expected. Re-running it appends new session lines to the shipped `super_optimized_training.log` provenance log. `inference_model.py` additionally requires you to set `input_folder` in `main()` to your own crops. The other training variants and the per-year extractors are kept verbatim with their original machine-specific paths and are not expected to run from a fresh clone. Note the data-split note above: `optimized_data/` is the exact (contaminated) training input of the deployed detector, so retraining reproduces the deployed model's setup rather than a held-out benchmark.

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
