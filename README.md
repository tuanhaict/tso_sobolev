# TREE-SLICED SOBOLEV IPM

## Requirements
To install the required python packages, run
```
conda env create --file=environment.yaml
conda activate twd
cd power_spherical
```


Install `power_spherical` package:
```bash
cd power_spherical
pip install .
```

You may need to install additional requirements. Detail may be found in each experiment. 

## Included Experiments
* Gradient flow on Spheres (in /spherical/gradient_flow) and Euclidean (/experiments/gradient-flow)
* Denoising Diffusion (/experiment/denoising-diffusion-gan)
* Text Topic modeling /experiment/topic-modelling
* Self Supervised Learning (/spherical/ssl)