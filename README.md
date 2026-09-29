<img src="misc/logo.png" height="150">

# CNN-based Touchscreen Cyrillic Handwriting Recognition

**MSc thesis project.** This repository holds the code for a convolutional neural network (CNN) that classifies handwritten Cyrillic letters drawn on touchscreens. The model is trained on the [CoMNIST](https://github.com/GregVial/CoMNIST) dataset and compared against the pretrained CoMNIST model released by the dataset's original author, Gregory Vial.

The repository is a fork of [GregVial/CoMNIST](https://github.com/GregVial/CoMNIST). The dataset, the letter-reading API in [api/](api/) and the pretrained baseline weights are his work (see [Credits](#credits)). The thesis work is the training and evaluation pipeline described below.

## Contents

- [Dataset](#dataset)
- [Model](#model)
- [Experiments and results](#experiments-and-results)
- [Comparison with the original CoMNIST model](#comparison-with-the-original-comnist-model)
- [Repository layout](#repository-layout)
- [Reproducing the results](#reproducing-the-results)
- [Credits](#credits)
- [License](#license)

## Dataset

[CoMNIST](https://github.com/GregVial/CoMNIST) (Cyrillic-oriented MNIST) is a set of handwritten letters that volunteers drew on touchscreens through a web page. The Cyrillic part used here ([images/Cyrillic.zip](images/Cyrillic.zip)) has **15,480 PNG images at 278×278 px** in **34 classes**: the 33 letters of the Russian alphabet plus `I`. The `I` class is the separate vertical stroke of `Ы`, which the original word reader recombines into `Ы`.

The images store the stroke in the alpha channel on a transparent background, so the training pipeline loads them as **RGBA**.

[train_test_split.py](train_test_split.py) splits the data per class (seed 123):

| Split      | Images | How it is built                                              |
|------------|-------:|--------------------------------------------------------------|
| Train      |  8,656 | 70% of each class, minus the validation part                 |
| Validation |  2,163 | 20% of the training folder, via Keras `validation_split=0.2` |
| Test       |  4,661 | the remaining 30% of each class, held out                    |

The classes are fairly balanced (about 290–400 training-folder images per letter). The exceptions are `I` (172) and `Ё` (240).

## Model

The model is a compact CNN built with TensorFlow / Keras:

```
Input 32×32×4 (RGBA)
Rescaling(1/255)
Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
Flatten
Dense(128, ReLU)
Dropout(0.5)
Dense(34)            # raw logits, softmax or sigmoid depending on the experiment
```

- **Trainable parameters:** 40,578
- **Optimizer:** Adam
- **Epochs:** 15
- **Hardware:** trained on a single NVIDIA GTX 1050 Ti

## Experiments and results

All variants share the architecture above and were evaluated on the same held-out test set (4,661 images). Precision, recall and F1 are macro-averaged over the 34 classes. The full table is in [metrics.csv](metrics.csv).

| Variant                                   | Accuracy | Precision | Recall |   F1 |
|-------------------------------------------|---------:|----------:|-------:|-----:|
| Base CNN, no dropout (raw logits, CCE)    |     0.87 |      0.88 |   0.87 | 0.87 |
| + Dropout, batch 32                       |     0.89 |      0.90 |   0.89 | 0.89 |
| + Dropout, batch 16                       |     0.89 |      0.90 |   0.89 | 0.89 |
| + Dropout, batch 64                       |     0.90 |      0.90 |   0.90 | 0.90 |
| + Dropout, softmax output + CCE           |     0.90 |      0.90 |   0.90 | 0.90 |
| + Dropout, sigmoid output + CCE           |     0.90 |      0.91 |   0.91 | 0.90 |
| **+ Dropout, softmax output + MSE loss**  | **0.91** |  **0.91** | **0.91** | **0.91** |

*CCE = categorical cross-entropy.*

Findings:

- **Dropout is the largest single improvement** (+2 points accuracy). It also narrows the gap between validation and test accuracy (from 9.5 to 6.5 points). It helped most on the visually similar `Щ` / `Ш` / `Ц` group.
- Batch size and the choice of output activation change the results by about one point.
- The best variant uses a softmax output trained with mean squared error.

## Comparison with the original CoMNIST model

The baseline is the pretrained Cyrillic model that Gregory Vial published with CoMNIST ([api/model.py](api/model.py), weights in [api/weights/comnist_keras_ru.hdf5](api/weights/comnist_keras_ru.hdf5)). It was run unchanged through its own preprocessing pipeline ([api/image_proc.py](api/image_proc.py)) on the same test images; see [model_test.ipynb](model_test.ipynb).

| Model                                        | Input        | Trainable params | Accuracy | Precision | Recall |   F1 |
|----------------------------------------------|--------------|-----------------:|---------:|----------:|-------:|-----:|
| Original CoMNIST CNN (G. Vial, pretrained)   | 32×32 gray   |       ~2.59 M    |   0.90   |    0.91   |  0.90  | 0.90 |
| This work: Dropout + softmax + MSE           | 32×32 RGBA   |      40.6 K      |   0.91   |    0.91   |  0.91  | 0.91 |

The model from this work reaches the same or slightly better scores with about **64× fewer parameters** than the original model.

## Repository layout

| Path                                         | Description                                                                 |
|----------------------------------------------|-----------------------------------------------------------------------------|
| [model_train.ipynb](model_train.ipynb)       | Thesis notebook: data loading, class distribution, training of all variants, confusion matrices, metrics |
| [model_test.ipynb](model_test.ipynb)         | Thesis notebook: evaluation of the original pretrained CoMNIST model on the test split |
| [model_train.html](model_train.html), [model_test.html](model_test.html) | Static exports of the two notebooks with all outputs |
| [code.pdf](code.pdf)                         | PDF export of the thesis code                                              |
| [train_test_split.py](train_test_split.py)   | Per-class 70/30 train/test split                                           |
| [metrics.csv](metrics.csv)                   | Macro metrics of every trained variant                                      |
| [api/](api/)                                 | Original CoMNIST letter/word-reading API and pretrained weights (G. Vial)  |
| [images/Cyrillic.zip](images/Cyrillic.zip)   | CoMNIST Cyrillic dataset                                                    |
| [misc/](misc/)                               | Original CoMNIST logo, presentation and contributor list                   |

## Reproducing the results

Requires Python 3.10 with TensorFlow, scikit-learn, NumPy, pandas, Matplotlib, seaborn and Pillow.

```bash
# 1. Extract the dataset (creates images/Cyrillic/<letter>/*.png)
unzip images/Cyrillic.zip -d images/

# 2. Build the train/test split (images/Cyrillic_train, images/Cyrillic_test)
python train_test_split.py

# 3. Run the notebooks from the repository root
jupyter notebook model_train.ipynb   # train and evaluate the new CNN variants
jupyter notebook model_test.ipynb    # evaluate the original CoMNIST model
```

The notebooks import `api.*` as a package, so start them from the repository root.

## Credits

- **Gregory Vial** created the [CoMNIST dataset](https://github.com/GregVial/CoMNIST), the letter-reading API and the **pretrained CNN used as the baseline** in this thesis. More background is on his [blog post about CoMNIST](http://ds.gregvi.al/2017/02/28/CoMNIST/).
- **Anna Migushina** built the crowd-sourcing web page ([github](https://github.com/migusta/coMNIST)) used to collect the images.
- **Everyone who drew letters** for the dataset: see the [contributors list](misc/contributors.md).
- CoMNIST logo by **Sophie Valentina**.

## License

The CoMNIST dataset and the original code are licensed under CC BY-SA 4.0:

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">CoMNIST</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/GregVial/CoMNIST" property="cc:attributionName" rel="cc:attributionURL">Gregory Vial</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

Because the license is ShareAlike, the work derived from it in this repository is distributed under the same license.
