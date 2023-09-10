# Image Generation Models (work in progress)

This repository provides implementations of several generative models applied to image data. These models are primed for understanding and experimenting with advanced image-based generative techniques.

The implemented models are sourced from the following research papers:
- [Score based model](https://arxiv.org/abs/2011.13456)
- [Diffusion model](https://arxiv.org/abs/2006.11239)
- [Critically-Damped Langevin Diffusion](https://arxiv.org/abs/2112.07068)
- [Stochastic interpolant](https://arxiv.org/abs/2303.08797)

---

## Table of Contents

- [Running the Models](#running-the-models)
- [Toy Datasets](#toy-datasets)
- [Image Datasets](#image-datasets)

---

## Running the Models

The configuration files can be found in the `config_files` directory. Use the following command to run a specific model based on the desired configuration:

```bash
python main.py --config_file "config_files/toy/score_toy_config"
```

Examples and further illustrations are provided in the **`notebook/`** directory.

---

## Toy Datasets

### Sampling from the models:

| **Diffusion / Score Based Model** | **Critical-damped Langevin** | **Stochastic Interpolant** |
|:--------------------------------:|:----------------------------:|:--------------------------:|
| ![Diffusion Score Image](docs/assets/toy_traj_score.gif) | ![Critical Damped Image](docs/assets/toy_traj_cld.gif) | ![Stochastic Image](docs/assets/toy_traj_stochastic_interpolant.gif) |
| ![Score Sample Image](docs/assets/toy_score_sample.png) | ![Langevin Sample Image](docs/assets/toy_cld_sample.png) | ![Interpolant Sample Image](docs/assets/toy_stochastic_interpolant_sample.png) |

---

## Image Datasets

### Sampling from the Fashion MNIST dataset:


| **Diffusion / Score Based Model** | **Critical-damped Langevin** | **Stochastic Interpolant** |
|:--------------------------------:|:----------------------------:|:--------------------------:|
| ![Diffusion MNIST Image](docs/assets/fm_traj_score.gif) | ![Langevin MNIST Image](docs/assets/fm_traj_cld.gif) | TODO |
| ![Diffusion MNIST Sample](docs/assets/fm_score_sample.png) | ![Langevin MNIST Sample](docs/assets/fm_cld_sample.png) | TODO |

---



