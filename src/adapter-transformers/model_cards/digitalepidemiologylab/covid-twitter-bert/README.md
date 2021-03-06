---
language: "en"
thumbnail: "https://raw.githubusercontent.com/digitalepidemiologylab/covid-twitter-bert/master/images/COVID-Twitter-BERT_small.png"
tags:
- Twitter
- COVID-19
license: "MIT"
---

# COVID-Twitter-BERT (CT-BERT) v1

:warning: _You may want to use the [v2 model](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2) which was trained on more recent data and yields better performance_ :warning: 


BERT-large-uncased model, pretrained on a corpus of messages from Twitter about COVID-19. Find more info on our [GitHub page](https://github.com/digitalepidemiologylab/covid-twitter-bert).

## Overview
This model was trained on 160M tweets collected between January 12 and April 16, 2020 containing at least one of the keywords "wuhan", "ncov", "coronavirus", "covid", or "sars-cov-2". These tweets were filtered and preprocessed to reach a final sample of 22.5M tweets (containing 40.7M sentences and 633M tokens) which were used for training.

This model was evaluated based on downstream classification tasks, but it could be used for any other NLP task which can leverage contextual embeddings. 

In order to achieve best results, make sure to use the same text preprocessing as we did for pretraining. This involves replacing user mentions, urls and emojis. You can find a script on our projects [GitHub repo](https://github.com/digitalepidemiologylab/covid-twitter-bert).

## Example usage
```python
tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
model = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
```

You can also use the model with the `pipeline` interface:

```python
from transformers import pipeline
import json

pipe = pipeline(task='fill-mask', model='digitalepidemiologylab/covid-twitter-bert-v2')
out = pipe(f"In places with a lot of people, it's a good idea to wear a {pipe.tokenizer.mask_token}")
print(json.dumps(out, indent=4))
[
    {   
        "sequence": "[CLS] in places with a lot of people, it's a good idea to wear a mask [SEP]",
        "score": 0.9959408044815063,
        "token": 7308,
        "token_str": "mask"
    },  
    ... 
]
```

## References
[1] Martin M??ller, Marcel Salat??, Per E Kummervold. "COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter" arXiv preprint arXiv:2005.07503 (2020).
