# Data Augmentation for Abstractive Query-Focused Multi-Document Summarization (AAAI 2021)

This is the implementation of the paper [Data Augmentation for Abstractive Query-Focused Multi-Document Summarization](https://arxiv.org/pdf/2103.01863.pdf).

## Prerequisites

- Python 3.6+
- [PyTorch 1.0] (http://pytorch.org/)
- Install all the required packages from requirements.txt file.
  ```
  pip install -r requirements.txt
  ```
- Download the processed datasets (in pytorch format) and setup some folders and repos by running the following command: 
  ```
  python setup.py
  ```
  if you face any issues in downloading the datasets with the above code (setup.py), directly download the datasets from here: [wikisum](https://drive.google.com/uc?id=1AnqeUpLkO9MR3PH0V8q32A6PEPDEZ0td), [wikisum-query](https://drive.google.com/uc?id=1RdX-t3pznnyaGyrswFubAfoo9S9w9K5d), [qmds-cnn](https://drive.google.com/uc?id=1KXsvfnK6s6cnYQzD8ZOkXPdA6r5-quPK), [qmds-cnn-query](https://drive.google.com/uc?id=12i_3dikeJLsOj-SQGPmc4w9Is7fB-hT-). Run the above code with the following argument to setup everything else except the datasets. 
  ```
  python setup.py --ignore_datasets
  ```
- If you face any issues with running ROUGE evaluation, checkout this [link](https://poojithansl7.wordpress.com/2018/08/04/setting-up-rouge/).

- Some codes are borrowed from: [hiersumm](https://github.com/nlpyang/hiersumm) and [ONMT](https://github.com/OpenNMT/OpenNMT-py).
  

## Usage

To train the model:
```
DATASET=[CNNDM/WIKI] MODEL_TYPE=[hier/he/order/query/heq/heo/hero] bash run_experiments.sh 
```
To test the model:
```
DATASET=[CNNDM/WIKI] MODEL_TYPE=[hier/he/order/query/heq/heo/hero] bash test.sh 
```

Few points to note: 
- Various model types (`MODEL_TYPE`):
  - hier: Baseline model (Hierarchical Transformers)
  - he: HS w/ Hierarchical Encodings
  - order: HS w/ Ordering Component 
  - query: HS w/ Query Encoding
  - heq: HS-Joint Model (Hierachical Encodings + Query Encoding)
  - heo: HS-Joint Model (Hierachical Encodings + Ordering Component)
  - hero: HS-Joint Model (all three components combined)
- We tested our models on Nvidia P-100s 16GB. Each experiments uses 4 GPUs. If you have fewer gpus or memory, set `BATCH_SIZE`, `VISIBLE_GPUS`, `ACCUM_COUNT` accordingly.
- data, vocab, and model paths are set to default locations. Set these variables if you want to use different paths.

## Reference

If you find this code helpful, please consider citing the following paper:

    @inproceedings{pasunuru2021data,
        title={Data Augmentation for Abstractive Query-Focused Multi-Document Summarization},
        author={Pasunuru, Ramakanth and Celikyilmaz, Asli and Galley, Michel and Xiong, Chenyan and Zhang, Yizhe and Bansal, Mohit and Gao, Jianfeng},
        booktitle={AAAI},
        year={2021}
    }
    
