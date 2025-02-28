# RTML_Assignment_4

# Fine-Tuning MAE for Image Classification  
**Coursework: Using Pretrained Masked Autoencoder for MNIST & CIFAR-10 Classification**  

---

## 🔹 1. Introduction  

In this experiment, we trained a **Masked Autoencoder (MAE)** on **MNIST**, using different **patch sizes and masking ratios** to reconstruct images. After training, we **fine-tuned the MAE's encoder** for classification tasks on **MNIST** and later **extended it to CIFAR-10**.  

### **Objectives:**  
- Evaluate the effect of **patch size & masking ratio** on MAE performance.  
- Use the **pretrained MAE encoder** for **classification tasks**.  
- Compare **MNIST** and **CIFAR-10** performance.  
- Discuss **potential improvements** for CIFAR-10 classification.  

---

## 🔹 2. Experimental Setup  

### **2.1. Datasets**  

| Dataset  | Images (Train/Test) | Classes       | Color Channels |
|----------|--------------------|--------------|---------------|
| MNIST    | 60,000 / 10,000    | Digits (0-9) | Grayscale (1) |
| CIFAR-10 | 50,000 / 10,000    | 10 Objects   | RGB (3)       |

---

### **2.2. Training Configuration**  

| Hyperparameter       | Value  |
|----------------------|--------|
| **Batch Size**      | 256    |
| **Optimizer**       | AdamW  |
| **Learning Rate**   | 3e-4   |
| **Weight Decay**    | 0.05   |
| **Epochs (MAE)**    | 10     |
| **Epochs (Classif.)** | 10    |
| **Patch Sizes**     | 2, 4, 7, 14 |
| **Masking Ratios**  | 0.3, 0.5, 0.7, 0.9 |

---

## 🔹 3. Training the Masked Autoencoder (MAE)  

The MAE was trained on **MNIST** using **different patch sizes and masking ratios**.  

### **3.1. MAE Training Results**  

#### **Training Loss Across Epochs**  

| Patch Size | Mask Ratio | Epoch 1  | Epoch 5  | Epoch 10  |
|------------|------------|---------|---------|----------|
| 2          | 0.3        | 0.4893  | 0.2661  | 0.2513   |
| 2          | 0.5        | 0.4355  | 0.2623  | 0.2522   |
| 7          | 0.5        | 0.6934  | 0.2671  | 0.2410   |
| 14         | 0.5        | 0.8252  | 0.3141  | **0.2322 (Best Training Loss)** |

#### **Validation Loss After Training**  

| Patch Size | Mask Ratio | Validation Loss |
|------------|------------|----------------|
| 7          | 0.5        | **0.2395 (Best Validation Loss)** |
| 14         | 0.5        | **0.2302 (Final Best Model)** |

**Final Best Model:** **Patch Size = 14, Mask Ratio = 0.5, Validation Loss = 0.2302**  
**Best Model Saved at:** `saved_eval/mae-mnist-best.pth`  

### **Best MAE Validation Image Results**
![image](https://github.com/user-attachments/assets/680d0374-f4d9-4d72-8851-bf46684228e7)


---

## 🔹 4. Fine-Tuning the MAE Encoder for Classification  

### **4.1. Modifying the MAE Encoder**  
- The **decoder** was **removed**, keeping only the **encoder**.  
- A **classification token (CLS)** was **added**.  
- A **linear classifier (MLP head)** was used to classify the latent representation.  

### **4.2. MNIST Classification Results**  

| Epoch | Training Accuracy (%) | Validation Accuracy (%) |
|--------|----------------------|------------------------|
| 1      | 82.3                 | 80.7                   |
| 5      | 98.3                 | 97.0                   |
| 10     | **99.2**              | **98.5**               |

**Final MNIST Classifier Best Predictions:**  
![image](https://github.com/user-attachments/assets/eae4b777-930a-4781-83df-0a37df5db568)   ![image](https://github.com/user-attachments/assets/dc171057-6de3-4e6f-86df-5d13ebca1661)



---

## 🔹 5. CIFAR-10 Classification  

Since **CIFAR-10 has 3 color channels (RGB)**, we modified the **first layer of MAE** to accept **3 channels** and used the same classifier architecture.  

### **5.1. CIFAR-10 Classification Results**  

| Epoch | Training Accuracy (%) | Validation Accuracy (%) |
|--------|----------------------|------------------------|
| 1      | 55.2                 | 51.0                   |
| 5      | 80.8                 | 75.6                   |
| 10     | **87.1**              | **82.9**               |

**Final CIFAR-10 Classifier Best Predictions:**  
![image](https://github.com/user-attachments/assets/9648cb49-ec71-4866-83bb-f71f9b266d54)


---

## 🔹 6. MNIST vs. CIFAR-10: Key Comparisons  

| Factor              | MNIST (Easy Task)  | CIFAR-10 (Harder Task) |
|--------------------|------------------|---------------------|
| **Image Type**     | Simple digits     | Complex objects    |
| **Color**         | Grayscale (1)    | RGB (3 channels)  |
| **MAE Transfer Learning** | **Very High (98.5%)** | **Moderate (82.9%)** |
| **Best Performance Reached** | **5 Epochs** | **10+ Epochs Required** |

---

## 🔹 7. Potential Improvements for CIFAR-10  

To **improve CIFAR-10 classification**, we can:  
**Use More Data Augmentations**: Random cropping, horizontal flipping, color jittering.  
**Train for More Epochs**: CIFAR-10 might need 30+ epochs.  
**Use a Larger ViT Model**: Increase transformer depth.  
**Use Pretrained Weights from Larger Datasets**: Train on **ImageNet** for better transfer learning.  

