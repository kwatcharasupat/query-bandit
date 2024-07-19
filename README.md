<div align="center">
	<img src="assets/banquet-logo.png">
</div>

# Banquet: A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems

Repository for **A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems** 
by Karn N. Watcharasupat and Alexander Lerch. [arXiv](https://arxiv.org/abs/2406.18747)

> Despite significant recent progress across multiple subtasks of audio source separation, few music source separation systems support separation beyond the four-stem vocals, drums, bass, and other (VDBO) setup. Of the very few current systems that support source separation beyond this setup, most continue to rely on an inflexible decoder setup that can only support a fixed pre-defined set of stems. Increasing stem support in these inflexible systems correspondingly requires increasing computational complexity, rendering extensions of these systems computationally infeasible for long-tail instruments. In this work, we propose Banquet, a system that allows source separation of multiple stems using just one decoder. A bandsplit source separation model is extended to work in a query-based setup in tandem with a music instrument recognition PaSST model. On the MoisesDB dataset, Banquet, at only 24.9 M trainable parameters, approached the performance level of the significantly more complex 6-stem Hybrid Transformer Demucs on VDBO stems and outperformed it on guitar and piano. The query-based setup allows for the separation of narrow instrument classes such as clean acoustic guitars, and can be successfully applied to the extraction of less common stems such as reeds and organs.

For the Cinematic Audio Source Separation model, Bandit, see [this repository](https://github.com/kwatcharasupat/bandit).


**Model weights and instruction for using it will come soon! (or latest by mid-August!)**

## Citation

```
@inproceedings{Watcharasupat2024Banquet,
  title = {A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems},
  booktitle = {To Appear in the Proceedings of the 25th International Society for Music Information Retrieval},
  author = {Watcharasupat, Karn N. and Lerch, Alexander},
  year = {2024},
  month = {nov},
  eprint = {2406.18747},
  address = {San Francisco, CA, USA},
}
```
