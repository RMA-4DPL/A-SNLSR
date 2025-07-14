# Baseline results with splits by tile

Tiles used for training and validation were split randomly accross all 6 scenes. The following models were trained using the "source" SWIR 1 (1.0µm to 1.7µm) spectrum and metrics were evaluated on the "target" SWIR 2 (1.7µm to 2.5µm) spectrum.

## Bicubic interpolation (no training)

Val/PSNR   31.8863371
Val/SSIM    0.7801298
Val/ERGAS   0.6436654
Val/SAM     5.0990154
Val/RMSE    0.0255194

## RCAMNetwork:
Val/PSNR     33.200301 ± 0.130642
Val/SSIM      0.783413 ± 0.023903
Val/ERGAS     0.821341 ± 0.041854
Val/SAM       5.271546 ± 0.309542
Val/RMSE      0.021958 ± 0.000327

## SNLSRNetwork
Val/PSNR     31.983362 ± 0.032614
Val/SSIM      0.778001 ± 0.000271
Val/ERGAS     0.915132 ± 0.021563
Val/SAM       5.213153 ± 0.011012
Val/RMSE      0.025254 ± 0.000095

## ESRT
Val/PSNR     33.434913 ± 0.226760
Val/SSIM      0.779092 ± 0.022898
Val/ERGAS     0.802153 ± 0.043375
Val/SAM       5.615214 ± 0.344349
Val/RMSE      0.021373 ± 0.000556

## SRFormer
Val/PSNR     33.332589 ± 0.538714
Val/SSIM      0.786018 ± 0.022191
Val/ERGAS     0.791078 ± 0.021017
Val/SAM       5.328742 ± 0.401011
Val/RMSE      0.025541 ± 0.001191
