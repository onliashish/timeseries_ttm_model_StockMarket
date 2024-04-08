---
license: cdla-permissive-2.0
---

# Model Card for TTM

TTM, also known as TinyTimeMixer, are compact pre-trained models for Time-Series Forecasting, open-sourced by IBM Research.
**With less than 1 Million parameters, TTM introduces the notion of the first-ever “tiny” pre-trained models for Time-Series Forecasting.** 

TTM outperforms several popular benchmarks demanding billions of parameters in zero-shot and few-shot forecasting. TTM is pre-trained on diverse public time-series datasets which 
can be easily fine-tuned for your target data. Refer to our [paper](https://arxiv.org/pdf/2401.03955.pdf) for more details.

**Note that zeroshot, fine-tuning and inference tasks using TTM can easily be executed in 1 GPU machine or in laptops too!!**


## Benchmark Highlights:

- TTM (with less than 1 Million parameters) outperforms the following popular Pre-trained SOTAs demanding several hundred Million to Billions of parameters:
  - *GPT4TS (NeurIPS 23) by 7-12% in few-shot forecasting.*
  - *LLMTime (NeurIPS 23) by 24% in zero-shot forecasting*.
  - *SimMTM (NeurIPS 23) by 17% in few-shot forecasting*.
  - *Time-LLM (ICLR 24) by 8% in few-shot(5%) forecasting*
  - *UniTime (WWW 24) by 27% in zero-shot forecasting.*
- Zero-shot results of TTM surpass the *few-shot results of many popular SOTA approaches* including
  PatchTST (ICLR 23), PatchTSMixer (KDD 23), TimesNet (ICLR 23), DLinear (AAAI 23) and FEDFormer (ICML 22).
- TTM (1024-96, released in this model card with 1M parameters) outperforms pre-trained MOIRAI-Small (14M parameters) by 10%, MOIRAI-Base (91M parameters) by 2% and
  MOIRAI-Large (311M parameters) by 3% on zero-shot forecasting (fl = 96). (TODO: add notebook)
- TTM quick fine-tuning also outperforms the hard statistical baselines (Statistical ensemble and S-Naive) in
  M4-hourly dataset which existing pretrained TS models are finding hard to outperform. (TODO: add notebook)
- TTM takes only a *few seconds for zeroshot/inference* and a *few minutes for finetuning* in 1 GPU machine, as
  opposed to long timing-requirements and heavy computing infra needs of other existing pretrained models.
  

## Model Description

TTM falls under the category of “focused pre-trained models”, wherein each pre-trained TTM is tailored for a particular forecasting 
setting (governed by the context length and forecast length). Instead of building one massive model supporting all forecasting settings, 
we opt for the approach of constructing smaller pre-trained models, each focusing on a specific forecasting setting, thereby 
yielding more accurate results. Furthermore, this approach ensures that our models remain extremely small and exceptionally fast, 
facilitating easy deployment without demanding a ton of resources. 

Hence, in this model card, we plan to release several pre-trained 
TTMs that can cater to many common forecasting settings in practice. Additionally, we have released our source code along with 
our pretraining scripts that users can utilize to pretrain models on their own. Pretraining TTMs is very easy and fast, taking 
only 3-6 hours using 6 A100 GPUs, as opposed to several days or weeks in traditional approaches.

## Model Releases (along with the branch name where the models are stored):

- 512-96: Given the last 512 time-points (i.e. context length), this model can forecast up to next 96 time-points (i.e. forecast length)
  in future. Recommended for hourly and minutely forecasts (Ex. resolutions 5 min, 10 min, 15 min, 1 hour, etc)  (branch name: main) 

- 1024-96: Given the last 1024 time-points (i.e. context length), this model can forecast up to next 96 time-points (i.e. forecast length)
  in future. Recommended for hourly and minutely forecasts (Ex. resolutions 5 min, 10 min, 15 min, 1 hour, etc) (branch name: 1024-96-v1) 

- Stay tuned for more models !

## Model Details

For more details on TTM architecture and benchmarks, refer to our [paper](https://arxiv.org/pdf/2401.03955.pdf).

TTM-1 currently supports 2 modes:

 - **Zeroshot forecasting**: Directly apply the pre-trained model on your target data to get an initial forecast (with no training).

 - **Finetuned forecasting**: Finetune the pre-trained model with a subset of your target data to further improve the forecast.

**Since, TTM models are extremely small and fast, it is practically very easy to finetune the model with your available target data in few minutes 
to get more accurate forecasts.**

The current release supports multivariate forecasting via both channel independence and channel-mixing approaches. 
Decoder Channel-Mixing can be enabled during fine-tuning for capturing strong channel-correlation patterns across 
time-series variates, a critical capability lacking in existing counterparts.

In addition, TTM also supports exogenous infusion and categorical data which is not released as part of this version. 
Stay tuned for these extended features.

## Recommended Use
1. Users have to externally standard scale their data before feeding it to the model (Refer to TSP, our data processing utility for data scaling.)
2. Enabling any upsampling or prepending zeros to virtually increase the context length is not recommended and will
   impact the model performance.
   
 
### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]


## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

## How to Get Started with the Model

[Point notebooks]

## Benchmarks

## Training Data

The TTM models were trained on a collection of datasets from the Monash Time Series Forecasting repository. The datasets used include:
 - Australian Electricity Demand: https://zenodo.org/records/4659727 
 - Australian Weather: https://zenodo.org/records/4654822 
 - Bitcoin dataset: https://zenodo.org/records/5122101 
 - KDD Cup 2018 dataset: https://zenodo.org/records/4656756 
 - London Smart Meters: https://zenodo.org/records/4656091 
 - Saugeen River Flow: https://zenodo.org/records/4656058
 - Solar Power: https://zenodo.org/records/4656027 
 - Sunspots: https://zenodo.org/records/4654722
 - Solar: https://zenodo.org/records/4656144 
 - US Births: https://zenodo.org/records/4656049 
 - Wind Farms Production data: https://zenodo.org/records/4654858 
 - Wind Power: https://zenodo.org/records/4656032


## Citation [optional]
Kindly cite the following paper, if you intend to use our model or its associated architectures/approaches in your 
work

**BibTeX:**

@article{ekambaram2024ttms,
  title={TTMs: Fast Multi-level Tiny Time Mixers for Improved Zero-shot and Few-shot Forecasting of Multivariate Time Series},
  author={Ekambaram, Vijay and Jati, Arindam and Nguyen, Nam H and Dayama, Pankaj and Reddy, Chandra and Gifford, Wesley M and Kalagnanam, Jayant},
  journal={arXiv preprint arXiv:2401.03955},
  year={2024}
}

**APA:**

Ekambaram, V., Jati, A., Nguyen, N. H., Dayama, P., Reddy, C., Gifford, W. M., & Kalagnanam, J. (2024). TTMs: Fast Multi-level Tiny Time Mixers for Improved Zero-shot and Few-shot Forecasting of Multivariate Time Series. arXiv preprint arXiv:2401.03955.


## Model Card Authors

Vijay Ekambaram, Arindam Jati, Pankaj Dayama, Nam H. Nguyen, Wesley Gifford and Jayant Kalagnanam

## Model Card Contact

[More Information Needed]

## IBM Public Repository Disclosure: 

All content in this repository including code has been provided by IBM under the associated 
open source software license and IBM is under no obligation to provide enhancements, 
updates, or support. IBM developers produced this code as an 
open source project (not as an IBM product), and IBM makes no assertions as to 
the level of quality nor security, and will not be maintaining this code going forward.