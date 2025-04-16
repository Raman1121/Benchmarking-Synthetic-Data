# Benchmarking Medical Synthetic Data
A Benchmark to conduct a detailed evaluation of the quality of medical synthetic data from Text-to-Image Models.

## Contents

- [Overview](#overview)
    - [Quantitative Analysis](#quantitative-analysis)
        - [Generative Quality Metrics](#generative-quality-metrics)
        - [Diversity and Mode Coverage Metrics](#diversity-and-mode-coverage-metrics)
    - [Qualitative Analysis](#qualitative-analysis)
- [Usage](#usage)
    - [Generating Synthetic Data](#generating-synthetic-data)
    - [Calculating Global Metrics](#overall-metrics)
    - [Calculating Conditional Metrics](#conditional-metrics)



# Overview
## Quantitative Analysis
The quantitative analysis is conducted at two levels:
- **Overall** (across all the pathologies in the MIMIC dataset)
- **Conditional** (Each metric calculated separately for each pathology)

### Metrics
#### Generative Quality Metrics
- **Fréchet inception distance (FID)**
- **Kernel Inception Distance (KID)**
- **Inception Distance (IS)**
- **Image-Text Alignment**

#### Diversity and Mode Coverage Metrics
- **Precision** (How many synthetic samples map to regions covered by real samples)
- **Recall** (How much of the real data distribution is covered by the synthetic data)
- **Diversity** (How well does the density of synthetic samples match the density of real samples)
- **Coverage** (How many different modes (clusters) of the real data are represented by the synthetic data)

## Qualitative Analysis
The qualitative analysis is conducted by evaluating performance on downstream tasks. We use the following tasks for this:
- Multi-Label Classification
- Radiology Report Generation

# Usage

## Quantitative Analysis
### Generating Synthetic Data 
- Using your favourite T2I Model, generate synthetic samples for each prompt in the test set (MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv).
- During generation, save the prompt and the corresponding synthetic image in a CSV file with columns '**prompt**' and '**img_savename**'. 
- Place the CSV file in **assets/CSV**
- Place the synthetic images in **assets/synthetic_images**

### Calculating Generative Quality Metrics
 

#### Global Metrics
```
cd Benchmarking-Synthetic-Data
./scripts/image_quality_metrics.sh
```
**Note:** Calculating FID, KID, etc can be memory intensive and might result in OOM. If this is your case, run the following:
```
cd Benchmarking-Synthetic-Data
./scripts/image_quality_metrics_memory_saving.sh
```
#### Conditional Metrics
```
cd Benchmarking-Synthetic-Data
./scripts/image_quality_metrics_conditional.sh
```

<div style="border: 1px solid #ccc; padding: 10px; background-color:#e7f3fe;">
  Results would be stored in <strong>Results/image_generation_metrics.csv</strong>
</div>

<div style="border: 1px solid #ccc; padding: 10px; background-color:rgb(236, 241, 240);">
  <strong>Tip:</strong> Provide the name of the model or the specific checkpoint in the argument <strong>EXTRA_INFO</strong>
</div>
