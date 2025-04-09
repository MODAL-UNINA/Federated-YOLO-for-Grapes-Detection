# Federated YOLO for Grapes Detection

> **University of Naples Federico II, Department of Mathematics and Applications**  
> *"R. Caccioppoli" Mathematical Modelling and Data Analysis (M.O.D.A.L.) Research Group*

This repository contains the code and resources for the Federated YOLO project, which aims to detect grapes using Federated Learning (FL) techniques. In particular, it offers a pre-trained model that has been trained within a federated framework, utilizing 10 distinct grapevine datasets, each associated with a client. The model is designed to enable evaluation on the corresponding test sets.


# Data Availability

The image datasets used in this project were sourced from various publicly available collections. To simulate a realistic Federated Learning (FL) scenario, we selected datasets capturing diverse environmental conditions. These include variations in the distance between the camera and the grapevines (e.g., close-up vs. far-range views) and lighting differences across scenes.

Below is the list of datasets used in the experiments:

- **[AI4Agriculture Grape Dataset](#)**  
  Contains 250 images with 5,076 labeled objects captured in a vineyard located in Ribera de Duero. It features Tempranillo grapevines and includes bounding box annotations.  
  🔗 Download: [https://datasetninja.com/ai4agriculture-grape-dataset](https://datasetninja.com/ai4agriculture-grape-dataset)

- **[wGrapeUNIPD-DL](#)**  
  Includes 373 images acquired from vertical shoot-position vineyards in six different Italian locations across various phenological stages.  
  🔗 Download: [https://zenodo.org/records/4066730](https://zenodo.org/records/4066730)

- **[Embrapa WGISD](#)**  
  The Wine Grape Instance Segmentation Dataset provides 300 annotated images (6,451 objects) from five different grape varieties. Captured under different lighting, focus, and phenological conditions.  
  🔗 Download: [https://zenodo.org/records/3361736](https://zenodo.org/records/3361736)

- **[Grapevine Bunch Detection Dataset](#)**  
  Provides images of grapevine bunches with YOLO-formatted annotations. From the original dataset, ~1000 images were selected to maintain class balance.  
  🔗 Download: [https://zenodo.org/records/7665520](https://zenodo.org/records/7665520)

- **[Grapes Dataset – Computer Vision Project](#)**  
  Contains 425 images of grapevine bunches suitable for object detection tasks.  
  🔗 Download: [https://universe.roboflow.com/grapes-xpaom/grapes-dataset](https://universe.roboflow.com/grapes-xpaom/grapes-dataset)

- **[Grapes (Unlabeled)](#)**  
  A collection of over 1000 high-quality unlabeled images of grapevines, used in this project through a semi-supervised labeling process.  
  🔗 Download: [https://www.kaggle.com/datasets/asadaliprofile/grapes](https://www.kaggle.com/datasets/asadaliprofile/grapes)

- **[Grape Counter Dataset](#)**  
  Includes 464 images of grape bunches annotated in YOLOv8 format.  
  🔗 Download: [https://universe.roboflow.com/florin-sacadat-cg7st/grape-counter](https://universe.roboflow.com/florin-sacadat-cg7st/grape-counter)

- **[Grape Multimodal Dataset (Green)](#)**  
  From this multimodal dataset, only the RGB images were selected for our study.It comprises 667 images of green grape bunches captured from various angles and lighting conditions.  
  🔗 Download: [https://www.scidb.cn/en/detail?dataSetId=84fa458dfc854fba8ce578b6d826b9c8](https://www.scidb.cn/en/detail?dataSetId=84fa458dfc854fba8ce578b6d826b9c8)

- **[Grape Multimodal Dataset (Purple)](#)**  
  From this multimodal dataset, only the RGB images were selected for our study.It contains 700 images of purple grape bunches under diverse viewpoints.  
  🔗 Download: [https://www.scidb.cn/en/detail?dataSetId=84fa458dfc854fba8ce578b6d826b9c8](https://www.scidb.cn/en/detail?dataSetId=84fa458dfc854fba8ce578b6d826b9c8)

- **[CERTH Grape Dataset](#)**  
  Offers 2,502 high-resolution images of ‘Crimson Seedless’ grapes, captured during the 2022–2023 growing season. A curated subset of 1,000 images was used in this study.  
  🔗 Download: [https://www.scidb.cn/en/detail?dataSetId=84fa458dfc854fba8ce578b6d826b9c8](https://www.scidb.cn/en/detail?dataSetId=84fa458dfc854fba8ce578b6d826b9c8)

We would like to thank the creators and researchers who made these datasets available, as it significantly contributed to the development of this work.


# Requirements

To set up the environment for global model evaluation, create a python environment using the provided `agriyolo.yml` file:
```sh
conda env create -f environment_yolo.yaml
conda activate agriyolo
```

# Execution Instructions


## Prepare Data
We provide our data in the folder `$your_folder/Datasets`.


## Data organization
Data used to train YOLO
```bash
Datasets
└── AI4Agriculture
    └── test
        ├── images
        └── labels
└── Certh grape dataset reduced
    └── test
        ├── images
        └── labels
└── Grape counter
    └── test
        ├── images
        └── labels
└── Grapes
    └── test
        ├── images
        └── labels
└── Grapes dataset
    └── test
        ├── images
        └── labels
└── Grapevine Bunch Detection Dataset
    └── test
        ├── images
        └── labels
└── Grape Multimodal Dataset (Green)
    └── test
        ├── images
        └── labels
└── Grape Multimodal Dataset (Purple)
    └── test
        ├── images
        └── labels
└── WGISD
    └── test
        ├── images
        └── labels
└── wGrapeUNIPD-DL
    └── test
        ├── images
        └── labels

```

To evaluate the global model on the test sets, execute the following command:
```sh
python yolo.py --eval 
            --batch_size 16\
            --device 0\
```

This command initiates the evaluation process, assessing the model's performance using the specified batch size and device. If you wish to perform predictions on the test images instead, use the `--no-eval` flag as shown below:
```sh
python yolo.py --no-eval 
            --batch_size 16\
            --device 0\
```
In this case, the model will generate predictions for the test images in the specified dataset and save the results accordingly. Note: The `--dataset` argument allows you to specify the dataset to be used, with 'wgisd' as the default value.

Feel free to modify these values based on your hardware or specific requirements for the project.
