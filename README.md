# Regularized_GAN
<br><br><br>

# Regularized_pix2pix in PyTorch

The code was based on the [pix2pix code](https://github.com/phillipi/pix2pix) and modify by [Marjorie Redon](https://github.com/RedonMarjorie)

**Regularized_pix2pix:  [Project](https://redon213.users.greyc.fr/) |  [Paper](https://openaccess.thecvf.com/content/ICCV2023W/e-Heritage/html/Redon_3D_Surface_Approximation_of_the_Entire_Bayeux_Tapestry_for_Improved_ICCVW_2023_paper.html) |  [Torch](https://github.com/RedonMarjorie/Regulerized_GAN)**

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/RedonMarjorie/Regulerized_GAN.git
```

- Install [PyTorch](http://pytorch.org and) 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.

### pix2pix train/test
- Put your RGB images inside the "Pano" folder

- Generate the results using
```bash
python3 test_pano.py --dataroot ./datasets/tapisserie --name tapisserie --model pix2pix --direction BtoA
```

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{redon,
  title = {{3D surface Approximation of the Entire Bayeux Tapestry for Improved Pedagogical Access}},
  author = {Redon, Marjorie and Pizenberg, Matthieu and Qu{\'e}au, Yvain and Elmoataz, Abderrahim},
  booktitle = {{4th ICCV Workshop on Electronic Cultural Heritage}},
  SERIES = {Proceedings of the IEEE International Conference on Computer Vision (ICCV) Workshops},
  year = {2023},
}
```

## Acknowledgments
Our code is inspired by [Pix2pix](https://github.com/phillipi/pix2pix).
