## Farmland Segmentation using Detectron 2

## Description

This project focuses on segmenting farmland areas using the Detectron 2 deep learning framework. It involves training a model to identify and segment farmland regions in images.

The dataset used for this project is available at: [https://universe.roboflow.com/farmland-wdqqo/detectron-2-dataset](https://universe.roboflow.com/farmland-wdqqo/detectron-2-dataset)[1].

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8
- PyTorch
- Detectron 2
- OpenCV

### Steps

1. Create a Conda environment:
```bash
conda create -n detectron_env python=3.8
```

2. Activate the environment:
```bash
conda activate detectron_env
```

3. Install PyTorch and related dependencies:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

4. Install Cython:
```bash
conda install cython
```

5. Clone the Detectron 2 repository:
```bash
git clone https://github.com/facebookresearch/detectron2.git
```

6. Install OpenCV:
```bash
pip install opencv-python
```

7. Install PyTorch with CUDA support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

8. Navigate to the Detectron 2 directory:
```bash
cd detectron2
```

9. Install Detectron 2:
```bash
pip install -e .
```

## Usage

### Getting Started

1. Activate the Conda environment:
```bash
conda activate detectron_env
```

2. Run the farmland segmentation script:
```bash
python farmland_segmentation.py
```

The script will train the model, evaluate it on the validation dataset, and perform inference on test images.

### Examples

The code includes examples of visualizing the training dataset, making predictions on test images, and evaluating the model's performance using the COCO evaluator.

## Features

- Farmland segmentation using Detectron 2
- Training and evaluation on a custom dataset
- Visualization of predictions and ground truth
- COCO evaluator for performance assessment

## Contributing

We welcome contributions to this project! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them
4. Push your changes to your forked repository
5. Submit a pull request

## License

This project is licensed under the [License Name] license. See the [LICENSE](LICENSE) file for more information.

Citations:
[1] https://universe.roboflow.com/farmland-wdqqo/detectron-2-dataset
