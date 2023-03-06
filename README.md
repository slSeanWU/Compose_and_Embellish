# Compose & Embellish: A Transformer-based Piano Generation System
Official PyTorch implementation of the paper:
 - Shih-Lun Wu and Yi-Hsuan Yang  
  **Compose & Embellish: Well-Structured Piano Performance Generation via A Two-Stage Approach**  
  _Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)_, 2023  
   [**Paper**](https://arxiv.org/abs/2209.08212) | [**Audio demo (Google Drive)**](https://bit.ly/comp_embel) | [**Model weights**](https://huggingface.co/slseanwu/compose-and-embellish-pop1k7)
  ![image](https://user-images.githubusercontent.com/31909739/223012496-1c0f3c3c-1aaf-404a-b6ae-a529e8d395d1.png)


## Prerequisites
  - **Python 3.8** and **CUDA 10.2** recommended
  - Install dependencies
    ```
    pip install -r requirements.txt
    pip install git+https://github.com/cifkao/fast-transformers.git@39e726864d1a279c9719d33a95868a4ea2fb5ac5
    ```
  - Download trained models from HuggingFace Hub (make sure you're in repository root directory)
    ```
    git clone https://huggingface.co/slseanwu/compose-and-embellish-pop1k7
    ```

## Generate piano performances (with our trained models)
  - Stage 1: generate lead sheets (i.e., melody + chord progression)
    ```
    python3 stage01_compose/inference.py \
      stage01_compose/config/pop1k7_finetune.yaml \
      generation/stage01 \
      20
    ```
    You'll have 20 lead sheets under `generation/stage01` after this step.  
  - Stage 2: generate full performances conditioned on Stage 1 lead sheets
    ```
    python3 stage02_embellish/inference.py \
      stage02_embellish/config/pop1k7_default.yaml \
      generation/stage01 \
      generation/stage02
    ```
    The `samp_**_2stage_samp**.mid` files under `generation/stage02` are the final results.
    
## Training (finetuning) models on _AILabs.tw Pop1K7_ dataset
  - Stage 1: lead sheet (i.e. "**Compose**") model
    ```
    python3 stage01_compose/train.py stage01_compose/config/pop1k7_finetune.yaml
    ```
  - Stage 2: performance (i.e. "**Embellish**") model
    ```
    python3 stage02_embellish/train.py stage02_embellish/config/pop1k7_default.yaml
    ```
Note that these two commands may be run in parallel.

## Training on custom datasets
If you'd like to experiment with your own datasets, we suggest that you
  - read our **dataloaders** ([stage 1](https://github.com/slSeanWU/Compose_and_Embellish/blob/main/stage01_compose/dataloader.py), [stage 2](https://github.com/slSeanWU/Compose_and_Embellish/blob/main/stage02_embellish/dataloader.py)) and `.pkl` files of our **processed datasets** ([stage 1](https://huggingface.co/slseanwu/compose-and-embellish-pop1k7/tree/main/datasets/stage01_compose/pop1k7_finetune), [stage 2](https://huggingface.co/slseanwu/compose-and-embellish-pop1k7/tree/main/datasets/stage02_embellish/pop1k7_leedsheet2midi)) to understand what the models receive as inputs
  - refer to [CP Transformer repo](https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md) for a general guide on converting audio/MIDI files to event-based representations
  - use [musical structure analyzer](https://github.com/Dsqvival/hierarchical-structure-analysis) to get required structure markings for our stage 1 models.

## Acknowledgements
We would like to thank the following people for their open-source implementations that paved the way for our work:
  - [**Performer (fast-transformers)**](https://github.com/cifkao/fast-transformers/tree/39e726864d1a279c9719d33a95868a4ea2fb5ac5): Angelos Katharopoulos ([@angeloskath](https://github.com/angeloskath)) and Ondřej Cífka ([@cifkao](https://github.com/cifkao))
  - [**Transformer w/ relative positional encoding**](https://github.com/kimiyoung/transformer-xl): Zhilin Yang ([@kimiyoung](https://github.com/kimiyoung))
  - [**Musical structure analysis**](https://github.com/Dsqvival/hierarchical-structure-analysis): Shuqi Dai ([@Dsqvival](https://github.com/Dsqvival))
  - [**LakhMIDI melody identification**](https://github.com/gulnazaki/lyrics-melody/tree/main/pre-processing): Thomas Melistas ([@gulnazaki](https://github.com/gulnazaki))
  - [**Skyline melody extraction**](https://github.com/wazenmai/MIDI-BERT/tree/CP/melody_extraction/skyline): Wen-Yi Hsiao ([@wayne391](https://github.com/wayne391)) and Yi-Hui Chou ([@sophia1488](https://github.com/sophia1488))

## BibTex
If this repo helps with your research, please consider citing:
```
@inproceedings{wu2023compembellish,
  title={{Compose \& Embellish}: Well-Structured Piano Performance Generation via A Two-Stage Approach},
  author={Wu, Shih-Lun and Yang, Yi-Hsuan},
  booktitle={Proc. Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  url={https://arxiv.org/pdf/2209.08212.pdf}
}
```
