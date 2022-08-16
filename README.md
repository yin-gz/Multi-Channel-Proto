## Cost-effective CNNs-based prototypical networks for few-shot relation classification across domains

Preparation
---

1.Download datasets

The original datasets should be put in `./data/origin/train` and `./data/origin/test` directory.

The Fewrel 2.0 datasets can be downloaded [here](https://competitions.codalab.org/competitions/27981#participate-get_data).

The TAC dataset is released via the Linguistic Data Consortium (LDC). Therefore, you can download TACRED from the [LDC TACRED webpage](https://catalog.ldc.upenn.edu/LDC2018T24). 

2. Generate parsed data

Run "./data/parse.py" to generate the sentences' dependency trees. Data after parsing should be put in `./data/origin_parse' directory.


3. Download pretrain checkpoint

Due to the large size, the pre-trained Glove files and BERT pretrain checkpoint are not included. You can download them [here](https://drive.google.com/drive/folders/1lXComU1HNd2dN4uU_SmUCpMosHbY5jF8?usp=sharing) to `./checkpoint` directory.

Training a Model
---

1. To train our models(MCE+SR), first use the following command to generate and select pseudo-labeled data

```shell
python train_demo.py --parse=True --save_ckpt=path_to_your_saved_model --cluster --M=2 --pseudo_pth=="train_wiki_and_pseudo_pubmed"
```

3. Then, re-train the instance encoder
```shell
python train_demo.py --parse=True --train="train_wiki_and_pseudo_pubmed" --save_ckpt=path_to_your_saved_model
```

Evaluation
---
After downloading the test dataset and put them to the right directory, use command to generate labels for the test dataset
```shell
python train_demo.py --load_ckpy=path_to_your_saved_model --only_test --test_online='test_pubmed_input'
```

Note: The model is not deterministic. All the experimental results presented in the paper are averaged across multiple runs.

## Other Training examples
1. To train proto-CNN, use command
```shell
python train_demo.py --model=proto --encoder=cnn --adv=None --save_ckpt=path_to_your_saved_model
```
2. To train BERT-PAIR, use command
```shell
python train_demo.py --model=pair --pair --encoder=bert --adv=None --hidden_size=768 --optim=adamw --lr=2e-5 --save_ckpt=path_to_your_saved_model
```

## Reference
 If you make use of this code or the GroupIM algorithm in your work, please cite the following paper:
```
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