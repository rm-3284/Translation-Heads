#!/usr/bin/env python
# coding: utf-8

# number of spaces, number of non-alphabets
# 
# no backprop option?
# 
# random sentences
# 
# different languages
# 

# identify useful heads
# get the mean of attention values 
# (just English sentences -> activation across all the sentences), 
# also max and min value
# Ablation and logits of the first word in another language
# # %%

# In[1]:


import os
import sys
from pathlib import Path

import pkg_resources

IN_COLAB = "google.colab" in sys.modules

# Install dependencies
installed_packages = [pkg.key for pkg in pkg_resources.working_set]
if "transformer-lens" not in installed_packages:
    get_ipython().run_line_magic('pip', 'install transformer_lens==2.11.0 einops eindex-callum jaxtyping git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python')


# In[2]:


import functools
import sys
from pathlib import Path
from typing import Callable

import circuitsvis as cv
from datasets import load_dataset
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

from helper import (
    dataset_clean,
    detect_diff_head,
    find_translation_token,
    loop_dict_to_sentence,
    make_mat_mean,
    plot_mean_attn,
    randomize_sentences,
    translation_attn_detector,
    translation_attn_detector_normalized,
    translation_attn_detector_mat,
)
from plotly_utils_local import hist, imshow, plot_comp_scores, plot_logit_attribution, plot_loss_difference

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# In[37]:
model: HookedTransformer = HookedTransformer.from_pretrained("gemma-2-2B")

# In[38]:
model.cfg

# 5 languages > 1000 sentences - Arabic, Japanese, Korean, French, German
# diagonal case with normalization
# control - random case (almost the same length)

# In[42]:
########################################################################
# English to French
ds_fr = load_dataset("Helsinki-NLP/opus-100", "en-fr")
fr_data = ds_fr['train']['translation'][:1000]
fr_data = dataset_clean(model, fr_data, ('en', 'fr'))
fr_data_random = randomize_sentences(model, fr_data, ("en", "fr"))
# In[44]:
fr_sentence_list = loop_dict_to_sentence(fr_data, ('en', 'fr'), ('EN', 'FR'))
fr_sentence_random = loop_dict_to_sentence(fr_data_random, ('en', 'fr'), ('EN', 'FR'))
mean_mat_fr = make_mat_mean(model, fr_sentence_list, 'FR')
mean_mat_rand_fr = make_mat_mean(model, fr_sentence_random, 'FR')

# **Reasons why there is no dominant translation head**
# - The sentences might not be suitable for this task (it is not word to word translation and you cannot capture the translation)
# - The model might not be powerful enough (Without specifying that this is a translation task, it gives me wrong output as seen in note.ipynb)
# - Attention detector is not accurate
# - threshold might be inappropriate
# - Depending on input, the role of heads could change

# In[46]:
########################################################################
# English to German

ds_de = load_dataset("Helsinki-NLP/opus-100", "de-en") # German
de_data = ds_de['train']['translation'][:1000]
de_data = dataset_clean(model, de_data, ('en', 'de'))
de_data_random = randomize_sentences(model, de_data, ("en", "de"))
# %%
de_sentence_list = loop_dict_to_sentence(de_data, ('en', 'de'), ('EN', 'DE'))
de_sentence_rand = loop_dict_to_sentence(de_data_random, ('en', 'de'), ('EN', 'DE'))
mean_mat_de = make_mat_mean(model, de_sentence_list, 'DE')
mean_mat_rand_de = make_mat_mean(model, de_sentence_rand, 'DE')

# In[52]:
########################################################################
# English to Japanese
ds_ja = load_dataset("Helsinki-NLP/opus-100", "en-ja")
ja_data = ds_ja['train']['translation'][:1000]
ja_data = dataset_clean(model, ja_data, ('en', 'ja'))
ja_data_random = randomize_sentences(model, ja_data, ("en", "ja"))
# %%
ja_sentence_list = loop_dict_to_sentence(ja_data, ('en', 'ja'), ('EN', 'JA'))
ja_sentence_rand = loop_dict_to_sentence(ja_data_random, ('en', 'ja'), ('EN', 'JA'))
mean_mat_ja = make_mat_mean(model, ja_sentence_list, 'JA')
mean_mat_rand_ja = make_mat_mean(model, ja_sentence_rand, 'JA')

# In[3]:
########################################################################
# English to Arabic
ds_ar = load_dataset("Helsinki-NLP/opus-100", "ar-en")
ar_data = ds_ar['train']['translation'][:1000]
ar_data = dataset_clean(model, ar_data, ('en', 'ar'))
ar_data_random = randomize_sentences(model, ar_data, ("en", "ar"))
# %%
ar_sentence_list = loop_dict_to_sentence(ar_data, ('en', 'ar'), ('EN', 'AR'))
ar_sentence_rand = loop_dict_to_sentence(ar_data_random, ('en', 'ar'), ('EN', 'AR'))
mean_mat_ar = make_mat_mean(model, ar_sentence_list, 'AR')
mean_mat_rand_ar = make_mat_mean(model, ar_sentence_rand, 'AR')

# %%
########################################################################
# English to Spanish
ds_es = load_dataset("Helsinki-NLP/opus-100", "en-es")
es_data = ds_es['train']['translation'][:1000]
es_data = dataset_clean(model, es_data, ('en', 'es'))
es_data_random = randomize_sentences(model, es_data, ("en", "es"))
# %%
es_sentence_list = loop_dict_to_sentence(es_data, ('en', 'es'), ('EN', 'ES'))
es_sentence_rand = loop_dict_to_sentence(es_data_random, ('en', 'es'), ('EN', 'ES'))
mean_mat_es = make_mat_mean(model, es_sentence_list, 'ES')
mean_mat_rand_es = make_mat_mean(model, es_sentence_rand, 'ES')

# %%
plot_mean_attn(mean_mat_fr, mean_mat_rand_fr, ('EN', 'FR'))
#%%
plot_mean_attn(mean_mat_de, mean_mat_rand_de, ('EN', 'DE'))
#%%
plot_mean_attn(mean_mat_ja, mean_mat_rand_ja, ('EN', 'JA'))
#%%
plot_mean_attn(mean_mat_ar, mean_mat_rand_ar, ('EN', 'AR'))
#%%
plot_mean_attn(mean_mat_es, mean_mat_rand_es, ('EN', 'ES'))

# %%

# threshold was chosen manually to account for the difference between languages

print(detect_diff_head(mean_mat_fr, mean_mat_rand_fr, 0.075))

print(detect_diff_head(mean_mat_de, mean_mat_rand_de, 0.075))

print(detect_diff_head(mean_mat_ja, mean_mat_rand_ja, 0.05))

print(detect_diff_head(mean_mat_ar, mean_mat_rand_ar, 0.075))

print(detect_diff_head(mean_mat_es, mean_mat_rand_es, 0.075))

# %%
########################################################################
# French to English
fr_reverse = loop_dict_to_sentence(fr_data, ('fr', 'en'), ('FR', 'EN'))
fr_reverse_random = loop_dict_to_sentence(fr_data_random, ('fr', 'en'), ('FR', 'EN'))
reverse_fr = make_mat_mean(model, fr_reverse, 'EN')
reverse_rand_fr = make_mat_mean(model, fr_reverse_random, 'EN')
plot_mean_attn(reverse_fr, reverse_rand_fr, lang=("FR", "EN"))
# %%
########################################################################
# German to English
de_reverse = loop_dict_to_sentence(de_data, ('de', 'en'), ('DE', 'EN'))
de_reverse_random = loop_dict_to_sentence(de_data_random, ('de', 'en'), ('DE', 'EN'))
reverse_de = make_mat_mean(model, de_reverse, 'EN')
reverse_rand_de = make_mat_mean(model, de_reverse_random, 'EN')
plot_mean_attn(reverse_de, reverse_rand_de, lang=("DE", "EN"))
# %%
########################################################################
# Japanese to English
ja_reverse = loop_dict_to_sentence(ja_data, ('ja', 'en'), ('JA', 'EN'))
ja_reverse_random = loop_dict_to_sentence(ja_data_random, ('ja', 'en'), ('JA', 'EN'))
reverse_ja = make_mat_mean(model, ja_reverse, 'EN')
reverse_rand_ja = make_mat_mean(model, ja_reverse_random, 'EN')
plot_mean_attn(reverse_ja, reverse_rand_ja, lang=("JA", "EN"))
# %%
########################################################################
# Arabic to English
ar_reverse = loop_dict_to_sentence(ar_data, ('ar', 'en'), ('AR', 'EN'))
ar_reverse_random = loop_dict_to_sentence(ar_data_random, ('ar', 'en'), ('AR', 'EN'))
reverse_ar = make_mat_mean(model, ar_reverse, 'EN')
reverse_rand_ar = make_mat_mean(model, ar_reverse_random, 'EN')
plot_mean_attn(reverse_ar, reverse_rand_ar, lang=("AR", "EN"))
# %%
########################################################################
# Spanish to English
es_reverse = loop_dict_to_sentence(es_data, ('es', 'en'), ('ES', 'EN'))
es_reverse_random = loop_dict_to_sentence(es_data_random, ('es', 'en'), ('ES', 'EN'))
reverse_es = make_mat_mean(model, es_reverse, 'EN')
reverse_rand_es = make_mat_mean(model, es_reverse_random, 'EN')
plot_mean_attn(reverse_es, reverse_rand_es, lang=("ES", "EN"))
# %%
########################################################################
# Chinese to French
ds_zh_fr = load_dataset("Helsinki-NLP/opus-100", "fr-zh")
zh_fr_data = ds_zh_fr['test']['translation'][:1000]
zh_fr_data = dataset_clean(model, zh_fr_data, ('fr', 'zh'))
zh_fr_data_random = randomize_sentences(model, zh_fr_data, ("zh", "fr"))
# %%
zh_fr_sentence_list = loop_dict_to_sentence(zh_fr_data, ('zh', 'fr'), ('ZH', 'FR'))
zh_fr_sentence_rand = loop_dict_to_sentence(zh_fr_data_random, ('zh', 'fr'), ('ZH', 'FR'))
mean_mat_zh_fr = make_mat_mean(model, zh_fr_sentence_list, 'FR')
mean_mat_rand_zh_fr = make_mat_mean(model, zh_fr_sentence_rand, 'FR')
plot_mean_attn(mean_mat_zh_fr, mean_mat_rand_zh_fr, ('ZH', 'FR'))
# %%
########################################################################
# Chinese to German
ds_zh_de = load_dataset("Helsinki-NLP/opus-100", "de-zh")
zh_de_data = ds_zh_de['test']['translation'][:1000]
zh_de_data = dataset_clean(model, zh_de_data, ('de', 'zh'))
zh_de_data_random = randomize_sentences(model, zh_de_data, ("zh", "de"))
# %%
zh_de_sentence_list = loop_dict_to_sentence(zh_de_data, ('zh', 'de'), ('ZH', 'DE'))
zh_de_sentence_rand = loop_dict_to_sentence(zh_de_data_random, ('zh', 'de'), ('ZH', 'DE'))
mean_mat_zh_de = make_mat_mean(model, zh_de_sentence_list, 'DE')
mean_mat_rand_zh_de = make_mat_mean(model, zh_de_sentence_rand, 'DE')
plot_mean_attn(mean_mat_zh_de, mean_mat_rand_zh_de, ('ZH', 'DE'))
# %%
########################################################################
# Chinese to English
ds_zh_en = load_dataset("Helsinki-NLP/opus-100", "en-zh")
zh_en_data = ds_zh_en['train']['translation'][:1000]
zh_en_data = dataset_clean(model, zh_en_data, ('en', 'zh'))
zh_en_data_random = randomize_sentences(model, zh_en_data, ("zh", "en"))
# %%
zh_en_sentence_list = loop_dict_to_sentence(zh_en_data, ('zh', 'en'), ('ZH', 'EN'))
zh_en_sentence_rand = loop_dict_to_sentence(zh_en_data_random, ('zh', 'en'), ('ZH', 'EN'))
mean_mat_zh_en = make_mat_mean(model, zh_en_sentence_list, 'EN')
mean_mat_rand_zh_en = make_mat_mean(model, zh_en_sentence_rand, 'EN')
plot_mean_attn(mean_mat_zh_en, mean_mat_rand_zh_en, ('ZH', 'EN'))
# %%
########################################################################
# ' ZH' will be tokenized as ' Z' and 'H'
# French to Chinese
zh_fr_reverse = loop_dict_to_sentence(zh_fr_data, ('fr', 'zh'), ('FR', 'ZH'))
zh_fr_reverse_random = loop_dict_to_sentence(zh_fr_data_random, ('fr', 'zh'), ('FR', 'ZH'))
reverse_zh_fr = make_mat_mean(model, zh_fr_reverse, 'Z')
reverse_rand_zh_fr = make_mat_mean(model, zh_fr_reverse_random, 'Z')
plot_mean_attn(reverse_zh_fr, reverse_rand_zh_fr, lang=("FR", "ZH"))
# %%
########################################################################
# German to Chinese
zh_de_reverse = loop_dict_to_sentence(zh_de_data, ('de', 'zh'), ('DE', 'ZH'))
zh_de_reverse_random = loop_dict_to_sentence(zh_de_data_random, ('de', 'zh'), ('DE', 'ZH'))
# %%
reverse_zh_de = make_mat_mean(model, zh_de_reverse, 'Z')
reverse_rand_zh_de = make_mat_mean(model, zh_de_reverse_random, 'Z')
plot_mean_attn(reverse_zh_de, reverse_rand_zh_de, lang=("DE", "ZH"))
# %%
########################################################################
# English to Chinese
zh_en_reverse = loop_dict_to_sentence(zh_en_data, ('en', 'zh'), ('EN', 'ZH'))
zh_en_reverse_random = loop_dict_to_sentence(zh_en_data_random, ('en', 'zh'), ('EN', 'ZH'))
reverse_zh_en = make_mat_mean(model, zh_en_reverse, 'Z')
reverse_rand_zh_en = make_mat_mean(model, zh_en_reverse_random, 'Z')
plot_mean_attn(reverse_zh_en, reverse_rand_zh_en, lang=("EN", "ZH"))

# %%
