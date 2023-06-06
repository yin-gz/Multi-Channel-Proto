## Cost-effective CNNs-based prototypical networks for few-shot relation classification across domains

This code is the official implementation of the paper: Cost-effective CNNs-based prototypical networks for few-shot relation classification across domains (https://doi.org/10.1016/j.knosys.2022.109470).

## Requirements
We used Python 3.9 and PyTorch 1.8.0.
You can install all requirements with:
```shell
pip install -r requirements.txt
```

Preparation
---

1.Download datasets

The original datasets should be put in `./data/origin/train` and `./data/origin/test` directory.

The Fewrel 2.0 datasets can be downloaded [here](https://competitions.codalab.org/competitions/27981#participate-get_data).

The TAC dataset is released via the Linguistic Data Consortium (LDC). Therefore, you can download TACRED from the [LDC TACRED webpage](https://catalog.ldc.upenn.edu/LDC2018T24) and turn it into N-way-K-shot format. 

2. Generate parsed data

Run `./data/parse.py` to generate the sentences' dependency trees. Data after parsing should be put in `./data/origin_parse' directory.


3. Download pretrain checkpoint

Due to the large size, the pre-trained Glove files and BERT pretrain checkpoint are not included. You can download them [here](https://drive.google.com/drive/folders/1lXComU1HNd2dN4uU_SmUCpMosHbY5jF8?usp=sharing) to `./checkpoint` directory.

Training a Model
---

1. To train our models(MCE+SR), first use the following command to generate and select pseudo-labeled data

```shell
python train_demo.py \
       --parse --cluster --M 2 --word_att\
       --adv "unsupervised_pubmed" \
       --pseudo_pth "train_wiki_and_pseudo_pubmed" \
       --save_ckpt {path_to_your_saved_model} 
```

2. Then, re-train the instance encoder
```shell
python train_demo.py \
       --train "train_wiki_and_pseudo_pubmed" \
       --parse --word_att\
       --adv "unsupervised_pubmed" \
       --save_ckpt {path_to_your_saved_model}
```

Evaluation
---
After downloading the test dataset and put them to the right directory, use command to generate labels for the test dataset
```shell
python train_demo.py \
       --parse --word_att\
       --load_ckpt {path_to_your_saved_model} \
       --only_test --test_online 'test_pubmed_input'
```

Note: The model is not deterministic. All the experimental results presented in the paper are averaged across multiple runs.

## Other Training Examples
- To train proto-CNN, use command
```shell
python train_demo.py \
       --model "proto" --encoder "cnn"
```
- To train BERT-PAIR, use command
```shell
python train_demo.py \
       --model "pair" --encoder "bert" --pair \
       --trainN 5 --N 5 --K 1 --Q 1 \
       --hidden_size 768 --lr 2e-5 --dropout 0.5 --optim adamw
```
- To train proto-bert, use command
```shell
python train_demo.py \
       --model "proto" --encoder "bert"\
       --trainN 5 --N 5 --K 5 --Q 5 \
       --hidden_size 768 --lr 2e-5  --dropout 0.5 --optim adamw
```

## Results Correction for Bert-based Methods on FewTAC
> [2023.6.6 update] In our original implementation, the BERT-based models couldn't correctly mark the entities and thus their performances on FewTAC was underestimated in the original paper.  We would like to offer our sincere apologies for the mistake. After correcting the codes and re-testing the models, the results are updated as follows.
### Correction Results on FewTAC
|                   | 5 way 5 shot | 5 way 10 shot | 10 way 5 shot | 10 way 10 shot |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| Proto+BERT (train_N=5, train_K=1)| 81.42	| 82.63 | 68.07	| 70.55 |
| Proto+BERT (train_N=5, train_K=5)| 83.77 | 85.71 | 72.5 | 75.23 |
| Proto+BERT+adv (train_N=5, train_K=5, λ=0.01)| 82.31 | 84.58 | 70.62 | 73.34 |
| Proto+Dafec (train_N=5, train_K=5, λ=0.01)| 83.81 | 85.82 | 72.28 | 74.90 |
| BERT-PAIR (train_N=5, train_K=1)| 84.16 | 85.67 | 73.18 | 75.23 |

#### Note:
- Adversarial training may not be always helpful and unstable on FewTAC, especially for BERT-based models. This may be due to the significant differences of sentence relation categories between TAC and WIKI.
- Proto+BERT-based methods would cost large GPU memory and much time, especially when setting train_K to 5. In comparision, our proposed methods are more cost-effective.




## Reference
If you find this work helpful in your research, please kindly consider citing the following paper. The bibtex are listed below:
```bibtex
@article{YIN2022109470,
title = {Cost-effective CNNs-based prototypical networks for few-shot relation classification across domains},
journal = {Knowledge-Based Systems},
volume = {253},
pages = {109470},
year = {2022},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2022.109470},
author = {Gongzhu Yin and Xing Wang and Hongli Zhang and Jinlin Wang}
}
```