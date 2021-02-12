# Automated coding for ICF Mobility activities
Automatically assigning 3-digit ICF codes to functional activity descriptions

This open-source software package implements a variety of methods for automatically coding descriptions of mobility activities in text documents, as described in the following paper:

+ D Newman-Griffis and E Fosler-Lussier, "[Automated Coding of Under-Studied Medical Concept Domains: Linking Physical Activity Reports to the International Classification of Functioning, Disability, and Health](https://arxiv.org/abs/2011.13978)". arXiv <i>2011.13978</i>.

## Setup/Installation

The included `makefile` provides pre-written commands for preprocessing, experimentation, and analysis of activity report data.

The `requirements.txt` file lists all required Python packages installable with pip. Just run
```
pip install -r requirements.txt
```
to install all packages.

Source code of two packages is required for generating BERT features; these packages ([BERT](https://github.com/google-research/bert) and [bert\_to\_hdf5](https://github.com/drgriffis/bert_to_hdf5)) are automatically downloaded by the `setup` target in the `makefile`.

## Package components

The processing pipeline in this package includes several primary elements, described here with reference to key code files. (For technical reference on script usage, see `makefile`)

- **Dataset preprocessing:** tokenization and formatting for analysis.
  + See ``preprocess_dataset_[spacy|bert]`` in `makefile`
- **Classification experiments using scikit-learn:** experiments under the classification paradigm, using k-Nearest Neighbors, Support Vector Machine, and Deep Neural Network models.
  + See ``run_classifier`` in `makefile`
- **Classification experiments using BERT fine-tuning:** adaptation of [BERT fine-tuning](https://github.com/google-research/bert) to the ICF coding case.
  + See ``utils/modified_BERT_run_classifier.py``
- **Candidate selection experiments:** experiments under the candidate selection paradigm
  + See ``experiments/candidate_selection``
- **Detailed analysis of experimental outputs:** performance and confusion analysis by ICF code
  + See ``analysis/per_code_performance.py``

## Demo data

The data used in the accompanying paper are not readily available due to patient confidentiality protections. Requests for information about the data may be directed to [julia.porcino@nih.gov](mailto:julia.porcino@nih.gov).

However, this package includes two tiny datasets for code demonstration purposes:

- ```data/demo_datasets/demo_labeled_dataset``` 5 short, synthetic clinical documents with mobility-related information. Text files are located in the `txt` subdirectory, and `csv` contains corresponding CSV files with standoff annotations.
- ```data/demo_datasets/demo_unlabeled_dataset``` 5 more short, synthetic clinical documents, only one of which contains mobility-related information. Text files are provided without corresponding annotations.

## Reference

If you use this software in your own work, please cite the following paper:
```
@article{newman-griffis2020automated,
  title={Automated Coding of Under-Studied Medical Concept Domains: Linking Physical Activity Reports to the International Classification of Functioning, Disability, and Health},
  author={Newman-Griffis, Denis and Fosler-Lussier, Eric},
  journal={arXiv preprint arXiv:2011.13978},
  year={2020}
}
```

## License

All source code, documentation, and data contained in this package are distributed under the terms in the LICENSE file (modified BSD).

<img src="https://cc.nih.gov/internet/general/images/NIH_CC_logo.png" />
