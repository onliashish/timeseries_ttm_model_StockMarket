---
license: apache-2.0
---

# TinyTimeMixer (TTM) Model Card

<p align="center" width="100%">
<img src="ttm_image.webp" width="600">
</p>

TinyTimeMixers (TTMs) are compact pre-trained models for Multivariate Time-Series Forecasting, open-sourced by IBM Research.
**With less than 1 Million parameters, TTM introduces the notion of the first-ever “tiny” pre-trained models for Time-Series Forecasting.** 

TTM outperforms several popular benchmarks demanding billions of parameters in zero-shot and few-shot forecasting. TTMs are lightweight 
forecasters, pre-trained on publicly available time series data with various augmentations. TTM provides state-of-the-art zero-shot forecasts and can easily be 
fine-tuned for multi-variate forecasts with just 5% of the training data to be competitive. Refer to our [paper](https://arxiv.org/pdf/2401.03955v5.pdf) for more details.


**The current open-source version supports point forecasting use-cases ranging from minutely to hourly resolutions 
(Ex. 10 min, 15 min, 1 hour, etc.)**

**Note that zeroshot, fine-tuning and inference tasks using TTM can easily be executed in 1 GPU machine or in laptops too!!**


**Recent updates:** We have developed more sophisticated variants of TTMs (TTM-B, TTM-E and TTM-A), featuring extended benchmarks that compare them with some of the latest models 
such as TimesFM, Moirai, Chronos, Lag-llama, and Moment. For full details, please refer to the latest version of our [paper](https://arxiv.org/pdf/2401.03955.pdf). 
Stay tuned for the release of the model weights for these newer variants.

## How to Get Started with the Model

- [colab](https://colab.research.google.com/github/IBM/tsfm/blob/tutorial/notebooks/tutorial/ttm_tutorial.ipynb)
- [Getting Started Notebook](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/ttm_getting_started.ipynb)
- [512-96 Benchmarks](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_benchmarking_512_96.ipynb)
- [1024-96 Benchmarks](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_benchmarking_1024_96.ipynb)
- Script for Finetuning with cross-channel correlation support - to be added soon


## Benchmark Highlights:

- TTM (with less than 1 Million parameters) outperforms the following popular Pre-trained SOTAs demanding several hundred Million to Billions of parameters [paper](https://arxiv.org/pdf/2401.03955v5.pdf):
  - *GPT4TS (NeurIPS 23) by 7-12% in few-shot forecasting*
  - *LLMTime (NeurIPS 23) by 24% in zero-shot forecasting*.
  - *SimMTM (NeurIPS 23) by 17% in few-shot forecasting*.
  - *Time-LLM (ICLR 24) by 2-8% in few-shot forecasting*
  - *UniTime (WWW 24) by 27% in zero-shot forecasting.*
- Zero-shot results of TTM surpass the *few-shot results of many popular SOTA approaches* including
  PatchTST (ICLR 23), PatchTSMixer (KDD 23), TimesNet (ICLR 23), DLinear (AAAI 23) and FEDFormer (ICML 22).
- TTM (1024-96, released in this model card with 1M parameters) outperforms pre-trained MOIRAI-Small (14M parameters) by 10%, MOIRAI-Base (91M parameters) by 2% and
  MOIRAI-Large (311M parameters) by 3% on zero-shot forecasting (horizon = 96). [[notebook]](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_benchmarking_1024_96.ipynb)
- TTM quick fine-tuning also outperforms the competitive statistical baselines (Statistical ensemble and S-Naive) in
  M4-hourly dataset which existing pretrained TS models are finding difficult to outperform. [[notebook]](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_m4_hourly.ipynb)
- TTM takes only a *few seconds for zeroshot/inference* and a *few minutes for finetuning* in 1 GPU machine, as
  opposed to long timing-requirements and heavy computing infra needs of other existing pre-trained models.



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

Each pre-trained model will be released in a different branch name in this model card. Kindly access the required model using our 
getting started [notebook](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/ttm_getting_started.ipynb) mentioning the branch name.

## Model Releases (along with the branch name where the models are stored):

- **512-96:** Given the last 512 time-points (i.e. context length), this model can forecast up to next 96 time-points (i.e. forecast length)
  in future. This model is targeted towards a forecasting setting of context length 512 and forecast length 96 and
  recommended for hourly and minutely resolutions (Ex. 10 min, 15 min, 1 hour, etc).   (branch name: main) 

- **1024-96:** Given the last 1024 time-points (i.e. context length), this model can forecast up to next 96 time-points (i.e. forecast length)
  in future. This model is targeted towards a long forecasting setting of context length 1024 and forecast length 96 and
  recommended for hourly and minutely resolutions (Ex. 10 min, 15 min, 1 hour, etc). (branch name: 1024-96-v1) 

- Stay tuned for more models !

## Model Details

For more details on TTM architecture and benchmarks, refer to our [paper](https://arxiv.org/pdf/2401.03955v5.pdf).

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
1. Users have to externally standard scale their data independently for every channel before feeding it to the model (Refer to [TSP](https://github.com/IBM/tsfm/blob/main/tsfm_public/toolkit/time_series_preprocessor.py), our data processing utility for data scaling.)
2. Enabling any upsampling or prepending zeros to virtually increase the context length for shorter-length datasets is not recommended and will
   impact the model performance. 
   
 
### Model Sources

- **Repository:** https://github.com/IBM/tsfm/tree/main/tsfm_public/models/tinytimemixer
- **Paper:** https://arxiv.org/pdf/2401.03955v5.pdf
- **Paper (Newer variants, extended benchmarks):** https://arxiv.org/pdf/2401.03955.pdf


## Uses

```
# Load Model from HF Model Hub mentioning the branch name in revision field

model = TinyTimeMixerForPrediction.from_pretrained(
                "https://huggingface.co/ibm/TTM", revision="main"
            ) 

# Do zeroshot
zeroshot_trainer = Trainer(
        model=model,
        args=zeroshot_forecast_args,
        )
    )

zeroshot_output = zeroshot_trainer.evaluate(dset_test)


# Freeze backbone and enable few-shot or finetuning:

# freeze backbone
for param in model.backbone.parameters():
  param.requires_grad = False

finetune_forecast_trainer = Trainer(
        model=model,
        args=finetune_forecast_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )
finetune_forecast_trainer.train()
fewshot_output = finetune_forecast_trainer.evaluate(dset_test)

```


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

```
@misc{ekambaram2024tiny,
      title={Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series}, 
      author={Vijay Ekambaram and Arindam Jati and Pankaj Dayama and Sumanta Mukherjee and Nam H. Nguyen and Wesley M. Gifford and Chandra Reddy and Jayant Kalagnanam},
      year={2024},
      eprint={2401.03955},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

**APA:**

Ekambaram, V., Jati, A., Dayama, P., Mukherjee, S., Nguyen, N. H., Gifford, W. M., … Kalagnanam, J. (2024). Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series. arXiv [Cs.LG]. Retrieved from http://arxiv.org/abs/2401.03955


## Model Card Authors

Vijay Ekambaram, Arindam Jati, Pankaj Dayama, Nam H. Nguyen, Wesley Gifford and Jayant Kalagnanam


## IBM Public Repository Disclosure: 

All content in this repository including code has been provided by IBM under the associated 
open source software license and IBM is under no obligation to provide enhancements, 
updates, or support. IBM developers produced this code as an 
open source project (not as an IBM product), and IBM makes no assertions as to 
the level of quality nor security, and will not be maintaining this code going forward.