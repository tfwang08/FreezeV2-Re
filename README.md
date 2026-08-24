# FreezeV2-Re

Paper-faithful reimplementation of **FreeZeV2 / FreeZeV2.1** for zero-shot 6D object pose estimation on the BOP benchmark.

The project intentionally keeps the codebase small and separates paper-specified behavior from reverse-engineered implementation choices. No pose-specific training or fine-tuning is used: DINOv2 and GeDi are frozen feature extractors.

Implementation work lives on the `reproduce-bop` branch until the first reproducible BOP baseline is ready.
