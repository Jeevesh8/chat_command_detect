# Analysing and Learning From a Chat Command Dataset

## The Data

The dataset is present in the [``data/``](https://github.com/Jeevesh8/chat_command_detect/tree/main/data) folder. Upon close analysis, a few points are revealed:

1. As data is generated from speech, we see that a lot of data is just repeated. That is the same commands(the same exact transcript) occur many times. Only 248 unique samples exist.

2. Moreover, the train and validation data differ only in the data order and are composed of exactly the same utterances.

3. Hence, we are in a low-data scenario. With only 248 chat commands available to us, with around 6 action + 14 object + 4 location labels to learn.

See notebook performing some basic EDA and samples from data and labels at [``notebooks/eda.ipynb``](https://github.com/Jeevesh8/chat_command_detect/blob/main/notebooks/eda.ipynb).

## Basic Models

We begin our analysis with some basic models like Bag-of-Words based Naive-Bayes Classsifiers, and just maximizing cosine distance from a known set of word embeddings in [``notebooks/baseline_models.ipynb``](https://github.com/Jeevesh8/chat_command_detect/blob/main/notebooks/baseline_models.ipynb).

We observe that Naive-Bayes gives an almost perfect fit for the data. This is a strong indicatioin that all the different kinds of labels are independent given the text.

## A Description of our Task

The task at hand is quite similar to the task of [intent detection and slot filling](http://nlpprogress.com/english/intent_detection_slot_filling.html), which has been studied widely. The intent detection problem is that of detecting the intent of a statement as one of a few, in a closed set of options.

While intent detection is clearly a sentence classification problem, it is not so clear what is the best form for slot filling. Sometimes
it can be cast as a sequence tagging problem, and sometimes as a classification problem.

In the case of a closed set of slots with small finite number of options for each slot, it would probably be better to cast it as
a classification problem; as language models generally perform better on them. We will follow this route for now, explore others later.

## Recurrent Models

We continue our analysis, by beginning to use recurrent models. To prepare the environment and run, see [``notebooks/run_models/run_rnn.ipynb``](https://github.com/Jeevesh8/chat_command_detect/blob/main/notebooks/run_models/run_rnn.ipynb). Or see logs on [``WandB``](https://wandb.ai/jeevesh8/chat_cmds).

We try different architectures and observe several phenomenon:

1. Model with single LSTM layers, fall in local minima, and stay stuck there. Try comparing the ``lstm_action`` and ``lstm_action_single_layer`` runs WandB.

2. Multi-Task LSTM's when trained with the same learning rate as single task LSTM's, fall into local minima again. Hence, we have to train with a lower learning rate, which leads to a much gradual decrease in loss(takes quite longer to train). Compare ``lstm_all_three``, ``lstm_all_three_longer`` and ``lstm_all_three_orig_params`` runs on WandB.

3. Single Task LSTM's can quickly learn their respective tasks.

Note that to learn semantics well in our model, we train the models on fixed embeddings from [FastText](https://github.com/facebookresearch/fastText).

## Transformer Models

Next, we try finetuning pre-trained transformer models. We try several pre-trained models: ``bert-base-uncased``, ``roberta``, ``albert``. All of them seem to learn in very few epochs compared to their recurrant counterparts.

The finetuned checkpoints are available for ``bert-base-uncased`` and the recurrent models too.

## Evaluation

We generate a sample test set by performing simple replacements like ``Turn->Blow`` and ``Chinese->Mandarin``. It is present in [``test_data.csv``](https://github.com/Jeevesh8/chat_command_detect/blob/main/test_data.csv). The result on this dataset for various trained models can be found in [``notebooks/run_models/test_results.md``](https://github.com/Jeevesh8/chat_command_detect/blob/main/notebooks/run_models/test_results.md).

To evaluate, we can use the same script and the same command, but with setting ``inference->run_infer`` as ``true`` in [``config.yaml``](https://github.com/Jeevesh8/chat_command_detect/blob/main/config.yaml). And providing the name of run from wandb, to load weights from in ``inference->run_name``. For example, ``jeevesh8/chat_cmds/k95jqc9b`` is name of [this run](https://wandb.ai/jeevesh8/chat_cmds/runs/k95jqc9b/).

## Future Work

1. Try [Stack Propagation](https://aclanthology.org/D19-1214.pdf) framework instead of our Multi-Task one.

2. Try using conversationally pretrained models like [ConveRT](https://aclanthology.org/2020.findings-emnlp.196.pdf) or [DialoGPT](https://paperswithcode.com/paper/a-bi-model-based-rnn-semantic-frame-parsing), some papers reported that they perform better on intent detection tasks.

3. [Universal Sentence Embeddings](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder) have been trained with conversations too, and may be useful.

4. Try solving same task in a harder setting, like that of [compositional generalization](https://arxiv.org/abs/1711.00350) tasks.

5. Try zero-shot setting using [Prompting Methods](https://arxiv.org/abs/2107.13586).
