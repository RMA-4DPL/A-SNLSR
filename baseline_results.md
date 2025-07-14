# Baseline results with splits by tile

Tiles used for training and validation were split randomly across all 6 scenes. The following models were trained using the "source" SWIR 1 (1.0µm to 1.7µm) spectrum and metrics were evaluated on the "target" SWIR 2 (1.7µm to 2.5µm) spectrum.

## Results

| Model | Val/PSNR | Val/SSIM | Val/ERGAS | Val/SAM | Val/RMSE |
|-------|----------|----------|-----------|---------|----------|
| Bicubic interpolation (no training) | 31.8863371 | 0.7801298 | 0.6436654 | 5.0990154 | 0.0255194 |
| RCAMNetwork | 33.200301 ± 0.130642 | 0.783413 ± 0.023903 | 0.821341 ± 0.041854 | 5.271546 ± 0.309542 | 0.021958 ± 0.000327 |
| SNLSRNetwork | 31.983362 ± 0.032614 | 0.778001 ± 0.000271 | 0.915132 ± 0.021563 | 5.213153 ± 0.011012 | 0.025254 ± 0.000095 |
| ESRT | 33.434913 ± 0.226760 | 0.779092 ± 0.022898 | 0.802153 ± 0.043375 | 5.615214 ± 0.344349 | 0.021373 ± 0.000556 |
| SRFormer | 33.332589 ± 0.538714 | 0.786018 ± 0.022191 | 0.791078 ± 0.021017 | 5.328742 ± 0.401011 | 0.025541 ± 0.001191 |
