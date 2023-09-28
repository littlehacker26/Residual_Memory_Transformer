# Controllable Text Generation with **R**esidual **M**emory **T**ransformer (RMT)

This repository contains code, data, checkpoints, and training and evaluation instructions for the paper **"Controllable Text Generation with Residual Memory Transformer".**

If you have any questions during the implementation, please leave us comments, we will help you solve it as soon as possible.


## Dependence

* **Environment**
  - Our code is built on the conda environment `python=3.9`.
* **Packages**
  - Please run `pip install -r requirements.txt` to install the requirements.

<!-- * **Datasets**
  - Please download the **datasets.zip (753M)** used in our experiments. [[Here is the anonymous download link]](xx)

  - Please download the **datasets.zip (753M)** used in our experiments. [[Here is the anonymous download link]](xx) -->

* **CLM Models**
  - Please download the GPT2 models from the `huggingface.co` community.
  
  |CLM Models|Parameters|Download Links|
  |:---|:---|:---|
  |gpt2-medium|355M|https://huggingface.co/gpt2-medium|
  |gpt2-large|774M|https://huggingface.co/gpt2-large|
  |gpt2-xl|1.5B|https://huggingface.co/gpt2-xl|


## Data Description

This part introduces the data (datasets.zip) used in our experiments.
* **Pre-training** data `data/wikipedia`.
* **Word inclusion** experimental data `data/word_include`.
* **Sentiment** experimental data
  - Discriminator checkpoint `data/sentiment/disc_check`
  - RMT's training data `data/sentiment/training`
  - Testing prompt data `data/sentiment/test`

## Checkpoint Description

This part describles the trained RMT checkpoints. All the RMT checkpoints are trained with 3 residual blocks (layers) and based on GPT2-large (If you need others RMT checkpoints based on gpt2-median/xl, please leave us a comment) .

|Checkpoints|Descriptions|
|:-------|:-------|
|**pretrained_check.ckpt**|The ***pre-trained*** RMT checkpoint|
|**commongen**|The finetuned RMT checkpoint on ***commonsence*** data without control length task.|
|**length_control**|The finetuned RMT checkpoint on ***commonsence*** data and the control length task.|
|**attribute**|The RMT finetuned checkpoint on ***sentiment*** control data.|


## RMT Pre-training Guidelines

This part introduces the RMT's pre-training process.

* Please run the following commands for pre-training:
  ```
  cd ./bash
  bash train_gpt_general_pretrain.sh
  ```

* Main pre-training arguments of configuration
  ```
  --pretrain_path       ## The pre-training corpus data from the Wikipedia.
  --pretrain_path_val   ## The validation data for pre-training from the Wikipedia
  --model_name_or_path  ## The path for the base CLM models (GPT2).
  --residual_layer      ## The residual layer (block) number of RMT.
  ```

## Word Inclusion & Length Control Guidelines

This part contains the RMT's training process.

* Please run the following commands for word inclusion training:
  ```
  cd ./bash
  bash train_rmt_commonsense.sh
  bash train_rmt_context_tuning.sh
  ```

<!-- - cd ./script
- bash train_rmt_commonsense.sh
- bash train_context_tuning.sh -->

* Main training arguments of configuration
  ```
  --train_path   ## The path for training data
  --dev_path    ## The path for validation data
  --test_path   ## The path for testing data
  --model_name_or_path    ## The path for the CLM models (GPT2)
  --length_control      ## whether to add length control constraints, if --generated_len should be added in evaluation process
  --out_dir   ## the output directory to save the checkpoint.
  ```
  P.S. After training process is finished, the process will log the testing results, and output a result file with `csv` format.

* Please run the following commands for word inclusion evaluation:
  ```
  cd ./bash
  bash eval_rmt_commonsense.sh
  bash eval_rmt_c2gen.sh
  ```

* Main evaluation arguments of configuration
  ```
  --test_path   ## The path for testing data
  --model_name_or_path    ## The path for the CLM models (GPT2)
  --generated_len   ## The length control for text generation
  --max_length    ## The maximum text generation length
  --check_point_load    ## The path for the trained RMT checkpoint
  --length_control      ## whether to add length control constraints, if we add length control task to world inclusions, 'generated_len' should be added in evaluation process
  --generated_len     ## required length control
  ```

  P.S. The process will output the genenrated text, which is saved with the csv format.



## Sentiment Control Guidelines

This part contains the training process of sentiment control for RMT.

* Please run the following commands for sentiment control training
  ```
  cd ./bash
  bash train_rmt_att.sh
  ```
* Main training arguments of configuration
  ```
  --train_path                  ## The path for training data  
  --model_name_or_path          ## The path for the CLM model (GPT2)
  --disc_embedding_checkpoint   ## The path for the trained discriminators
  --ranking_scope ## Configure the size of re-ranked candidate tokens, which is defined in DisCup paper
  --top_p         ## Configure the size of re-ranked candidate tokens using top-p
  --out_dir                    ## The output directory to save the RMT checkpoint
  ```

* Please run the following commands for sentiment control text generation:
  ```
  cd ./bash
  bash eval_rmt_att.sh
  ```

* Main generation arguments of configuration
  - The test path contains prompts for RMT, which includes `negative_prompts.jsonl`, `neutral_prompts.jsonl` and `positive_prompts.jsonl`.

  ```
  --test_path   ## The path for testing data
  --model_name_or_path    ## The path for the CLM model (GPT2)
  --ranking_scope   ## Configure the size of re-ranked candidate tokens, which is defined in DisCup paper
  -- target_att     ##  determines which sentiments to control
  --check_point_load  ## The path for the trained RMT checkpoint.
  ```

  P.S. The process will generate the sentiment-controlled text, and the results are saved with `csv` format.

* Sentiment control evaluation
  - Please refer to the Jupyter script `sentiment_classifier.ipynb`, and evaluate the correctness, PPL, dist-1/2/3 results of the generated `csv` file.
