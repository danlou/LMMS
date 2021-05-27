# Language Modelling Makes Sense (LMMS)

This repository includes the code related to the ["LMMS Reloaded: Transformer-based Sense Embeddings for Disambiguation and Beyond"](https://arxiv.org/abs/2105.12449) paper.

If you're interested in code for the original LMMS paper from [ACL 2019](https://www.aclweb.org/anthology/P19-1569/), click here to move to the [LMMS_ACL19](https://github.com/danlou/LMMS/tree/LMMS_ACL19) branch.

This code is designed to use the [transformers](https://github.com/huggingface/transformers) package (v3.0.2), and the [fairseq](https://github.com/pytorch/fairseq) package (v0.9.0, only for RoBERTa models, more details in the paper).

## Table of Contents

- [Installation](#installation)
- [Download Sense Embeddings](#download-sense-embeddings)
- [Create Sense Embeddings](#create-sense-embeddings)
- [Evaluation](#evaluation)
- [Demos](#demo)
- [References](#references)

## Installation

### Prepare Environment

This project was developed on Python 3.6.5 from Anaconda distribution v4.6.2. As such, the pip requirements assume you already have packages that are included with Anaconda (numpy, etc.).
After cloning the repository, we recommend creating and activating a new environment to avoid any conflicts with existing installations in your system:

```bash
$ git clone https://github.com/danlou/LMMS.git
$ cd LMMS
$ conda create -n LMMS python=3.6.5
$ conda activate LMMS
# $ conda deactivate  # to exit environment when done with project
```

### Additional Packages

To install additional packages used by this project run:

```bash
pip install -r requirements.txt
```

The WordNet package for NLTK isn't installed by pip, but we can install it easily with:

```bash
$ python -c "import nltk; nltk.download('wordnet')"
```

### External Data

If you want to evaluate the sense embeddings on WSD or USM, you need the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/).

```bash
$ cd external/wsd_eval  # from repo home
$ wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
$ unzip WSD_Evaluation_Framework.zip
```

For evaluation on the [WiC](https://pilehvar.github.io/wic/) dataset:

```bash
$ cd external/wic  # from repo home
$ wget https://pilehvar.github.io/wic/package/WiC_dataset.zip
$ unzip WiC_dataset.zip
```

Details about downloading GWCS and our WordNet subset of SID will be added soon.

If you want to represent embeddings using annotations from [UWA](http://danlou.github.io/uwa/), you must download SemCor+UWA10 from this [link](https://drive.google.com/open?id=1qUmApNs0TPI4RPjXW2aZzIfoLIZia0Nc), extract the .zip, and place the folder in external/uwa/.


## Download Sense Embeddings

You can download the main LMMS-SP embeddings we produced for the paper from the links below.

These sense embeddings should be used with the Transformer models of the same model name.

Tasks comparing or combining LMMS-SP embeddings with contextual embeddings need to also use the corresponding sets of layer weights in data/weights/ (specific to each **S**ense **P**rofile).

We distribute sense embeddings as '.txt' files, in the standard GloVe format.

Place downloaded sense embeddings in data/vectors/<model_name>/.

### bert-large-cased
- LMMS SP-WSD: [sensekeys (X.X GB)](https://drive.google.com/TODO); [synsets (X.X GB)](https://drive.google.com/TODO)
- LMMS SP-USM: [sensekeys (X.X GB)](https://drive.google.com/TODO); [synsets (X.X GB)](https://drive.google.com/TODO); [synsets-300d (X.X GB)](https://drive.google.com/TODO)

### xlnet-large-cased
- LMMS SP-WSD: [sensekeys (X.X GB)](https://drive.google.com/TODO); [synsets (X.X GB)](https://drive.google.com/TODO)
- LMMS SP-USM: [sensekeys (X.X GB)](https://drive.google.com/TODO); [synsets (X.X GB)](https://drive.google.com/TODO); [synsets-300d (X.X GB)](https://drive.google.com/TODO)

### roberta-large
- LMMS SP-WSD: [sensekeys (X.X GB)](https://drive.google.com/TODO); [synsets (X.X GB)](https://drive.google.com/TODO)
- LMMS SP-USM: [sensekeys (X.X GB)](https://drive.google.com/TODO); [synsets (X.X GB)](https://drive.google.com/TODO); [synsets-300d (X.X GB)](https://drive.google.com/TODO)

### albert-xxlarge-v2
- LMMS SP-WSD: [sensekeys (2.4->7.9 GB)](https://drive.google.com/file/d/1JE6fccyFGCZCJ-YzbW_mPdLFB0i5Npk7/view?usp=sharing); [synsets (1.4->4.5 GB)](https://drive.google.com/file/d/1fKhPrVR305SfIQz6yjaePyIBecfV35bM/view?usp=sharing)
- LMMS SP-USM: [sensekeys (2.4->7.9 GB)](https://drive.google.com/file/d/18unQKiYynJPtiyBnGQcveN2Ah17BhJMf/view?usp=sharing); [synsets (1.4->4.5 GB)](https://drive.google.com/file/d/1r1vlV42WmM_Z01ktdNppYZQ4m2ArAggj/view?usp=sharing); [synsets-300d (0.1->0.3 GB)](https://drive.google.com/file/d/17wEspQpoZmuSd1f_NWZ973vj7LfJcQ_8/view?usp=sharing)


## Create Sense Embeddings

The creation of LMMS-SP sense embeddings involves a series of steps that have corresponding scripts.

Below you'll find usage descriptions for all the scripts along with the exact command to run in order to replicate the results in the paper (for albert-xxlarge-v2, as an example).

Assumes layer weights have already been determined for each sense profile. The [create_sense_weights.py](https://github.com/danlou/LMMS/blob/master/scripts/embed_annotations.py) script can be used to convert layer performance to weights.

### 1. [embed_annotations.py](https://github.com/danlou/LMMS/blob/master/scripts/embed_annotations.py) - Bootstrap sense embeddings from annotated corpora

Usage description.

```bash
$ python scripts/embed_annotations.py -h
usage: embed_annotations.py [-h] [-nlm_id NLM_ID]
                            [-sense_level {synset,sensekey}]
                            [-weights_path WEIGHTS_PATH]
                            [-eval_fw_path EVAL_FW_PATH] -dataset
                            {semcor,semcor_uwa10} [-batch_size BATCH_SIZE]
                            [-max_seq_len MAX_SEQ_LEN]
                            [-subword_op {mean,first,sum}] [-layers LAYERS]
                            [-layer_op {mean,max,sum,concat,ws}]
                            [-max_instances MAX_INSTANCES] -out_path OUT_PATH

Create sense embeddings from annotated corpora.

optional arguments:
  -h, --help            show this help message and exit
  -nlm_id NLM_ID        HF Transfomers model name (default: bert-large-cased)
  -sense_level {synset,sensekey}
                        Representation Level (default: sensekey)
  -weights_path WEIGHTS_PATH
                        Path to layer weights (default: )
  -eval_fw_path EVAL_FW_PATH
                        Path to WSD Evaluation Framework (default:
                        external/wsd_eval/WSD_Evaluation_Framework/)
  -dataset {semcor,semcor_uwa10}
                        Name of dataset (default: semcor)
  -batch_size BATCH_SIZE
                        Batch size (default: 16)
  -max_seq_len MAX_SEQ_LEN
                        Maximum sequence length (default: 512)
  -subword_op {mean,first,sum}
                        Subword Reconstruction Strategy (default: mean)
  -layers LAYERS        Relevant NLM layers (default: -1 -2 -3 -4)
  -layer_op {mean,max,sum,concat,ws}
                        Operation to combine layers (default: sum)
  -max_instances MAX_INSTANCES
                        Maximum number of examples for each sense (default:
                        inf)
  -out_path OUT_PATH    Path to resulting vector set (default: None)
```

Example usage:

```bash
$ python scripts/embed_annotations.py -nlm_id albert-xxlarge-v2 -sense_level sensekey -dataset semcor_uwa10 -weights_path data/weights/lmms-sp-wsd.albert-xxlarge-v2.weights.txt -layer_op ws -out_path data/vectors/sc_uwa10-sp-wsd.albert-xxlarge-v2.vectors.txt
```

To represent synsets instead of sensekeys, you may use the option '-sense_level synset'.

### 2. [extend_sensekeys.py](https://github.com/danlou/LMMS/blob/master/scripts/extend_sensekeys.py) - Propagate supervised representations (from annotations) through WordNet

Usage description.

```bash
$ python scripts/extend_sensekeys.py -h
usage: extend_sensekeys.py [-h] -sup_sv_path SUP_SV_PATH
                           [-ext_mode {synset,hypernym,lexname}] -out_path
                           OUT_PATH

Propagates supervised sense embeddings through WordNet.

optional arguments:
  -h, --help            show this help message and exit
  -sup_sv_path SUP_SV_PATH
                        Path to supervised sense vectors
  -ext_mode {synset,hypernym,lexname}
                        Max abstraction level
  -out_path OUT_PATH    Path to resulting extended vector set
```

Example usage:

```bash
python scripts/extend_sensekeys.py -sup_sv_path data/vectors/sc_uwa10-sp-wsd.albert-xxlarge-v2.vectors.txt -ext_mode lexname -out_path data/vectors/sc_uwa10-extended-sp-wsd.albert-xxlarge-v2.vectors.txt
```

To extend synsets instead of sensekeys, use the [extend_synsets.py](https://github.com/danlou/LMMS/blob/master/scripts/extend_synsets.py) script in a similar fashion.

### 3. [embed_glosses.py](https://github.com/danlou/LMMS/blob/master/scripts/embed_glosses.py) - Create sense embeddings based on WordNet's glosses and lemmas

Usage description.

```bash
$ python scripts/embed_glosses.py -h
usage: embed_glosses.py [-h] [-nlm_id NLM_ID] [-sense_level {synset,sensekey}]
                        [-subword_op {mean,first,sum}] [-layers LAYERS]
                        [-layer_op {mean,sum,concat,ws}]
                        [-weights_path WEIGHTS_PATH] [-batch_size BATCH_SIZE]
                        [-max_seq_len MAX_SEQ_LEN] -out_path OUT_PATH

Creates sense embeddings based on glosses and lemmas.

optional arguments:
  -h, --help            show this help message and exit
  -nlm_id NLM_ID        HF Transfomers model name
  -sense_level {synset,sensekey}
                        Representation Level
  -subword_op {mean,first,sum}
                        Subword Reconstruction Strategy
  -layers LAYERS        Relevant NLM layers
  -layer_op {mean,sum,concat,ws}
                        Operation to combine layers
  -weights_path WEIGHTS_PATH
                        Path to layer weights
  -batch_size BATCH_SIZE
                        Batch size
  -max_seq_len MAX_SEQ_LEN
                        Maximum sequence length
  -out_path OUT_PATH    Path to resulting vector set
```

Example usage:

```bash
$ python scripts/embed_glosses.py -nlm_id albert-xxlarge-v2 -sense_level sensekey -weights_path data/weights/lmms-sp-wsd.albert-xxlarge-v2.weights.txt -layer_op ws -out_path data/vectors/glosses-sp-wsd.albert-xxlarge-v2.vectors.txt
```

To represent synsets instead of sensekeys, you may use the option '-sense_level synset'.

For a better understanding of what strings we're actually composing to generate these sense embeddings, here are a few examples:

| Sensekey (sk) | Embedded String (sk's lemma, all lemmas, tokenized gloss) |
|:-------------:|:---------------------------------------------------------:|
|    earth%1:17:00::     | earth - Earth , earth , world , globe - the 3rd planet from the sun ; the planet we live on    |
|    globe%1:17:00::     | globe - Earth , earth , world , globe - the 3rd planet from the sun ; the planet we live on    |
|    disturb%2:37:00::   | disturb - disturb , upset , trouble - move deeply                                              |


### 4. [merge_avg.py](https://github.com/danlou/LMMS/blob/master/scripts/merge_avg.py) - Merging gloss and extended representations

Usage description.

```bash
$ python scripts/merge_avg.py -h
usage: merge_avg.py [-h] -v1_path V1_PATH -v2_path V2_PATH [-v3_path V3_PATH]
                    -out_path OUT_PATH

Averages and normalizes vector .txt files.

optional arguments:
  -h, --help          show this help message and exit
  -v1_path V1_PATH    Path to vector set 1
  -v2_path V2_PATH    Path to vector set 2
  -v3_path V3_PATH    Path to vector set 3. Missing vectors are imputated from
                      v2 (optional)
  -out_path OUT_PATH  Path to resulting vector set
```

Example usage:

```bash
$ python scripts/embed_glosses.py -v1_path data/vectors/sc_uwa10-extended-sp-wsd.albert-xxlarge-v2.vectors.txt -v2_path data/vectors/glosses-sp-wsd.albert-xxlarge-v2.vectors.txt -out_path data/vectors/lmms-sp-wsd.albert-xxlarge-v2.vectors.txt
```

## Evaluation

Each of the 5 tasks tackled in the paper has its own evaluation script in evaluation/.

We refer to the start of each evaluation script for example usage and more details.


## Demos

For easier application on downstream tasks, we also prepared demonstration files showcasing barebones applications of LMMS-SP for disambiguation and matching using WordNet.

- [demo_disambiguation.py](demo_disambiguation.py): Loads a Transformer model, LMMS SP-WSD sense embeddings, and spaCy (for lemmatization and POS-tagging) and applies them to disambiguate particular word in an example sentence.
- [demo_matching.py](demo_matching.py): Loads a Transformer model and LMMS SP-USM sense embeddings, and applies them to match sensekeys and synsets particular word/span in an example sentence.


## References

### Under Review

Current version featuring Sense Profiles, probing analysis, and extensive evaluation. Under review, you may reference preprint below.

```
@misc{loureiro2021lmms,
      title={LMMS Reloaded: Transformer-based Sense Embeddings for Disambiguation and Beyond}, 
      author={Daniel Loureiro and Alípio Mário Jorge and Jose Camacho-Collados},
      year={2021},
      eprint={2105.12449},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### ACL 2019

The original LMMS paper ([ACL Anthology](https://www.aclweb.org/anthology/P19-1569/), [arXiv](https://arxiv.org/abs/1906.10007)).

```
@inproceedings{loureiro-jorge-2019-language,
    title = "Language Modelling Makes Sense: Propagating Representations through {W}ord{N}et for Full-Coverage Word Sense Disambiguation",
    author = "Loureiro, Daniel  and
      Jorge, Al{\'\i}pio",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1569",
    doi = "10.18653/v1/P19-1569",
    pages = "5682--5691"
}
```


### EMNLP 2020

Where we improve LMMS sense embeddings using automatic annotations for unambiguous words (UWA corpus) ([ACL Anthology](https://www.aclweb.org/anthology/2020.emnlp-main.283), [arXiv](https://arxiv.org/abs/2004.14325)).

```
@inproceedings{loureiro-camacho-collados-2020-dont,
    title = "Don{'}t Neglect the Obvious: On the Role of Unambiguous Words in Word Sense Disambiguation",
    author = "Loureiro, Daniel  and
      Camacho-Collados, Jose",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.283",
    doi = "10.18653/v1/2020.emnlp-main.283",
    pages = "3514--3520"
}
```


### SemDeep-5 at IJCAI 2019

Application of LMMS for the Word-in-Context (WiC) Challenge ([ACL Anthology](https://www.aclweb.org/anthology/W19-5801/), [arXiv](https://arxiv.org/abs/1906.10002)).

```
@inproceedings{loureiro-jorge-2019-liaad,
    title = "{LIAAD} at {S}em{D}eep-5 Challenge: Word-in-Context ({W}i{C})",
    author = "Loureiro, Daniel  and
      Jorge, Al{\'\i}pio",
    booktitle = "Proceedings of the 5th Workshop on Semantic Deep Learning (SemDeep-5)",
    month = aug,
    year = "2019",
    address = "Macau, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-5801",
    pages = "1--5",
}
```
