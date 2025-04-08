# Benchmarking-Synthetic-Data
A Benchmark to evaluate the quality of medical synthetic data from T2I Models

## Calculating Generative Quality Metrics

#### FID, KID, Inception Score and Image-Text Alignment

- Place the synthetic images in the 'assets/synthetic_images' folder
- Place the CSV containing path to synthetic images in the 'assets/CSV' folder
- Run the following command. Results would be stored in 'Results/image_generation_metrics.csv'

```
cd Benchmarking-Synthetic-Data
./scripts/img_quality_metrics.sh
``` 
