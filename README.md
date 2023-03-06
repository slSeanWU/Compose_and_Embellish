# Compose & Embellish: A Transformer-based Piano Generation System
Official PyTorch implementation of the paper:
 - Shih-Lun Wu and Yi-Hsuan Yang  
  **Compose & Embellish: Well-Structured Piano Performance Generation via A Two-Stage Approach**  
  _Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)_, 2023  
   [**Paper**](https://arxiv.org/abs/2209.08212) | [**Audio demo (Google Drive)**](https://bit.ly/comp_embel) | [**Model weights**](https://huggingface.co/slseanwu/compose-and-embellish-pop1k7)


## Acknowledgements
We would like to thank the following people for their open-source implementations that paved the way for our work:
  - [**Performer (fast-transformers)**](https://github.com/cifkao/fast-transformers/tree/39e726864d1a279c9719d33a95868a4ea2fb5ac5): Angelos Katharopoulos ([@angeloskath](https://github.com/angeloskath)) and Ondřej Cífka ([@cifkao](https://github.com/cifkao))
  - [**Transformer w/ relative positional encoding**](https://github.com/kimiyoung/transformer-xl): Zhilin Yang ([@kimiyoung](https://github.com/kimiyoung))
  - [**Musical structure analysis**](https://github.com/Dsqvival/hierarchical-structure-analysis): Shuqi Dai ([@Dsqvival](https://github.com/Dsqvival))
  - [**LakhMIDI melody identification**](https://github.com/gulnazaki/lyrics-melody/tree/main/pre-processing): Thomas Melistas ([@gulnazaki](https://github.com/gulnazaki))
  - [**Skyline melody extraction**](https://github.com/wazenmai/MIDI-BERT/tree/CP/melody_extraction/skyline): Wen-Yi Hsiao ([@wayne391](https://github.com/wayne391)) and Yi-Hui Chou ([@sophia1488](https://github.com/sophia1488))

## BibTex
If this repo is helpful for your research, please consider citing:
```
@inproceedings{wu2023compembellish,
  title={{Compose \& Embellish}: Well-Structured Piano Performance Generation via A Two-Stage Approach},
  author={Wu, Shih-Lun and Yang, Yi-Hsuan},
  booktitle={Proc. Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  url={https://arxiv.org/pdf/2209.08212.pdf}
}
```
