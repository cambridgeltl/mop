# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TF mBART model, originally from fairseq."""
from .configuration_mbart import MBartConfig
from .file_utils import add_start_docstrings
from .modeling_tf_bart import BART_START_DOCSTRING, TFBartForConditionalGeneration
from .utils import logging


_CONFIG_FOR_DOC = "MBartConfig"

START_DOCSTRING = BART_START_DOCSTRING.replace(
    "inherits from :class:`~transformers.TFPreTrainedModel`",
    "inherits from :class:`~transformers.TFBartForConditionalGeneration`",
).replace("BartConfig", _CONFIG_FOR_DOC)


logger = logging.get_logger(__name__)


@add_start_docstrings("mBART (multilingual BART) model for machine translation", START_DOCSTRING)
class TFMBartForConditionalGeneration(TFBartForConditionalGeneration):
    config_class = MBartConfig
    # All the code is in src/transformers/modeling_tf_bart.py
