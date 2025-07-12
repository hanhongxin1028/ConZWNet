# ConZWNet

This repository contains the implementation of ConZWNet. The code includes modules for data augmentation, data loading, model training, and evaluating robustness and discriminability.

## 📄 Paper

We published our work in the journal *Journal of Information Security and Applications*:

**Title**:  
**"ConZWNet: A contrastive learning-based zero-watermarking network for high robustness and distinguishability"**

**Authors**:  
Deyu Tong, Hongxin Han, Can Li, Fengting Wang, Weilong Kong, Na Ren

**Link**:  
👉 [ScienceDirect - Read the Paper](https://www.sciencedirect.com/science/article/pii/S2214212625001760?via%3Dihub)

**Abstract**:  
> Zero-watermarking is an effective solution for image copyright protection without altering the original content. However, current deep learning-based methods suffer from two key limitations. First, most feature extraction networks, originally designed for classification, lack robust feature learning essential for resisting attacks. Second, conventional methods seldom incorporate the generated watermark back into training, missing opportunities to further optimize the model. To address these issues, we propose ConZWNet, a two-stage framework that integrates contrastive learning with feedback-driven zero-watermark generation. In the first stage, we use ConvNeXt to learn invariant, attack-resistant features via contrastive learning on weak–strong augmentation. In the second stage, a residual network coupled with a Multi-Layer Perceptron (MLP) fuses features from host and copyright images to produce a latent zero-watermark, which is then verified by an MLP-based copyright identification network. This feedback loop optimizes feature fusion and transforms zero-watermark generation into a self-supervised process. Extensive experiments demonstrate that ConZWNet achieves state-of-the-art robustness against various attacks while ensuring high distinguishability among host images and copyrights. Ablation studies confirm the effectiveness of components, including two-stage architecture, contrastive learning, weak–strong augmentation, and copyright identification network. 

**BibTeX**:
```bibtex
@article{TONG2025104139,
title = {ConZWNet: A contrastive learning-based zero-watermarking network for high robustness and distinguishability},
journal = {Journal of Information Security and Applications},
volume = {93},
pages = {104139},
year = {2025},
issn = {2214-2126},
doi = {https://doi.org/10.1016/j.jisa.2025.104139},
url = {https://www.sciencedirect.com/science/article/pii/S2214212625001760},
author = {Deyu Tong and Hongxin Han and Can Li and Fengting Wang and Weilong Kong and Na Ren},
}


## 1. introduction
Our **ConZWNet** network has achieved an advanced level in zero-watermarking technology. We are the first to propose the application of contrastive learning and copyright label discrimination in the generation of zero-watermarks. The **robustness** test results show NC > **0.95**, and the **discriminability** test results show NC < **0.66** (for detailed experimental results, refer to Section 4).

Below is the architecture diagram of our network:
![ConZWNet_Overview](https://github.com/user-attachments/assets/197a9297-cb62-4444-bf78-1814619e856c)






## 2. Requirements

Key packages:

- Python = 3.12.8
- pytorch-cuda = 12.1

To install the required dependencies, please run:

```bash
conda create --name ConZWNet python=3.12.8
conda activate ConZWNet
conda install --file requirements.txt
```





## 3. Dataset

We used MiniImageNet as the host images for network training. You can find it on AI Studio at [MiniImageNet Link](https://aistudio.baidu.com/datasetdetail/167270).

We also used 200 school logo images as copyright images, which can be found in the [Google Drive Link](https://drive.google.com/drive/folders/11aKxaRsBGTz_jFRd3YgL3Gs6WqNCSANU?usp=drive_link)

The dataset should be organized as follows:

```plaintext
/data
  /copyright
  /images
  dog.jpg
  nufe.jpg
  test.csv
  train.csv
  val.csv
```

> Tips:
>
> * MiniImageNet is divided into the training, validation, and test sets using three CSV files. 
>
> * The copyright labels are obtained from the names of the copyright images.





## 4. Train the ConZWNet model

We trained the model using four 22GB RTX 2080 Ti GPUs. Our model training is divided into two phases: Phase 1 trains the robust feature extractor, and Phase 2 trains the complete ConZWNet model. The hyperparameters are defined as follows:

- **Learning Rate**: 0.001
- **Batch Size**: 256
- **Number of Epochs**: 200
- **Optimizer**: Adamax
- **Weight Decay**: 1e-4
- **temperature**: 0.5

In the [Google Drive Link](https://drive.google.com/drive/folders/1xEMRNOXadGWxobL58XOa22JuNR6zhkWA?usp=drive_link), you can also find the trained model weight files.

## 5. Results 

The model achieved the following performance on the test dataset: 

-  **Robustness**: 

| Attack Type                     | NC Value |
|---------------------------------|----------|
| Rotation (180°)                 | 0.9936   |
| Salt & Pepper Noise (0.1)       | 0.9542   |
| Gaussian Noise (0.1)            | 0.9991   |
| Random Crop (75%)               | 0.9547   |
| Corner Crop(Left-Bottom, 12.5%) | 0.9970   |
| Gaussian Filtering (11x11)      | 0.9981   |
| Median Filtering (11x11)        | 0.9965   |
| JPEG Compression (Q=20)         | 0.9919   |

* **discriminability**: 

| Host Image             | Copyright Image          | Averaged NC |
| ---------------------- | ------------------------ |-------------|
| 12,000 different hosts | Nufe                     | 0.6582      |
| Dog                   | 200 different copyrights | 0.4875      | 




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

