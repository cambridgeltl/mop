# Adapter Training

This section describes some examples on training different types of adapter modules in Transformer models.
The presented training scripts are only slightly modified from the original [examples by Huggingface](https://huggingface.co/transformers/examples.html).
To run the scripts, make sure you have the latest version of the repository and have installed some additional requirements:

```
git clone https://github.com/adapter-hub/adapter-transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

## Train a Task Adapter

Training a task adapter module on a dataset only requires minor modifications from training the full model. Suppose we have an existing script for training a Transformer model, here we will use HuggingFace's [run_glue.py](https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/text-classification/run_glue.py) example script for training on the GLUE dataset.

In our example, we replaced the built-in `AutoModelForSequenceClassification` class with the `AutoModelWithHeads` class introduced by `adapter-transformers` (learn more about prediction heads [here](prediction_heads.md)). Therefore, the model instantiation changed to:

```python
model = AutoModelWithHeads.from_pretrained(
        model_args.model_name_or_path,
        config=config,
)
model.add_classification_head(data_args.task_name, num_labels=num_labels)
```

Compared to fine-tuning the full model, there is only one significant adaptation we have to make: adding a new adapter module and activating it.

```python
# task adapter - only add if not existing
if task_name not in model.config.adapters.adapter_list(AdapterType.text_task):
    # resolve the adapter config
    adapter_config = AdapterConfig.load(
        adapter_args.adapter_config,
        non_linearity=adapter_args.adapter_non_linearity,
        reduction_factor=adapter_args.adapter_reduction_factor,
    )
    # add a new adapter
    model.add_adapter(
        task_name,
        AdapterType.text_task
        config=adapter_config
    )
# Enable adapter training
model.train_adapter([task_name])
```

```eval_rst
.. important::
    The most crucial step when training an adapter module is to freeze all weights in the model except for those of the
    adapter. In the previous snippet, this is achieved by calling the ``train_adapter()`` method which disables training
    of all weights outside the task adapter. In case you want to unfreeze all model weights later on, you can use
    ``freeze_model(False)``.
```

Besides this, we only have to make sure that the task adapter and prediction head are activated so that they are used in every forward pass. There are two ways to specify the adapter modules to be used. Either we can pass the parameter `adapter_names` in every call to `model.forward()`, or we can set the adapters to be used by default beforehand:

```python
model.set_active_adapters(task_name)
```

The rest of the training procedure does not require any further changes in code.

You can find the full version of the modified training script for GLUE at [run_glue_alt.py](https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/text-classification/run_glue_alt.py) in the `examples` folder of our repository.
We also adapted [various other example scripts](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples) (e.g. `run_glue.py`, `run_multiple_choice.py`, `run_squad.py`, ...) to support adapter training.

To start adapter training on a GLUE task, you can run something similar to:

```
export GLUE_DIR=/path/to/glue
export TASK_NAME=MNLI

python run_glue_alt.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 10.0 \
  --output_dir /tmp/$TASK_NAME \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer
```

The important flag here is `--train_adapter` which switches from fine-tuning the full model to training an adapter module for the given GLUE task.

```eval_rst
.. tip::
    Adapter weights are usually initialized randomly. That is why we require a higher learning rate. We have found that a default adapter learning rate of ``1e-4`` works well for most settings.
```

```eval_rst
.. tip::
    Depending on your data set size you might also need to train longer than usual. To avoid overfitting you can evaluating the adapters after each epoch on the development set and only save the best model.
```

## Train a Language Adapter

Training a language adapter is equally straightforward as training a task adapter. Similarly to the steps for task adapters
described above, we add a language adapter module to an existing model training script. Here, we modified the
[run_language_modeling.py](https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/contrib/legacy/run_language_modeling.py)
script by adding the following code:

```python
# check if language adapter already exists, otherwise add it
if language not in model.config.adapters.adapter_list(AdapterType.text_lang):
    # resolve the adapter config
    adapter_config = AdapterConfig.load(
        adapter_args.adapter_config,
        non_linearity=adapter_args.adapter_non_linearity,
        reduction_factor=adapter_args.adapter_reduction_factor,
    )
    model.add_adapter(language, AdapterType.text_lang, config=adapter_config)
# Freeze all model weights except of those of this adapter & use this adapter in every forward pass
model.train_adapter([language])
```

Training a language adapter on BERT then may look like the following:

```
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --language en \
    --train_adapter \
    --adapter_config pfeiffer
```

## Train AdapterFusion

We provide an example for training _AdapterFusion_ ([Pfeiffer et al., 2020](https://arxiv.org/pdf/2005.00247)) on the GLUE dataset: [run_fusion_glue.py](https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/text-classification/run_fusion_glue.py). You can adapt this script to train AdapterFusion with different pre-trained adapters on your own dataset.

```eval_rst
.. important::
    AdapterFusion on a target task is trained in a second training stage, after independently training adapters on individual tasks.
    When setting up a fusion architecture on your model, make sure to load the pre-trained adapter modules to be fused using ``model.load_adapter()`` before adding a fusion layer.
    For more on AdapterFusion, also refer to `Pfeiffer et al., 2020 <https://arxiv.org/pdf/2005.00247>`_.
```

To start fusion training on SST-2 as target task, you can run something like the following:

```
export GLUE_DIR=/path/to/glue
export TASK_NAME=SST-2

python run_fusion_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 10.0 \
  --output_dir /tmp/$TASK_NAME \
  --overwrite_output_dir
```
