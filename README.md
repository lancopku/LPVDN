# LPVDN

This repo implements the **LPVDN** (Locality Preserving Variational Discriminative Network) proposed in the paper 
**Learning Robust Representation for Clustering through Locality Preserving Variational Discriminative Network**

## Usage (Pytorch 1.5)

To train LPVDN on MNIST dataset with default setting, simply do:

```
python main.py
```

The expected results should be:

|  | ACC | NMI | ARI |
| :----: | :----: | :----: | :----: |
| LPVDN | 97.13&plusmn;0.21 | 92.76&plusmn;0.35 | 93.79&plusmn;0.42 |

