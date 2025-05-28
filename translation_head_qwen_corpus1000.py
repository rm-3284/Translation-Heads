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

model: HookedTransformer = HookedTransformer.from_pretrained("Qwen/Qwen2.5-3B")


# In[38]:


model.cfg

# 5 languages > 1000 sentences - Arabic, Japanese, Korean, French, German
# diagonal case with normalization
# control - random case (almost the same length)

# In[42]:
########################################################################
# French

ds_fr = load_dataset("Helsinki-NLP/opus-100", "en-fr")


# In[43]:


fr_data = ds_fr['train']['translation'][:1000]

fr_data = dataset_clean(model, fr_data, ('en', 'fr'))

# %%

fr_data_random = randomize_sentences(model, fr_data, ("en", "fr"))

# In[44]:


fr_sentence_list = loop_dict_to_sentence(fr_data, ('en', 'fr'), ('EN', 'FR'))

# %%

fr_sentence_random = loop_dict_to_sentence(fr_data_random, ('en', 'fr'), ('EN', 'FR'))


# %%

mean_mat_fr = make_mat_mean(model, fr_sentence_list, 'FR')

# %%

mean_mat_rand_fr = make_mat_mean(model, fr_sentence_random, 'FR')

# **Reasons why there is no dominant translation head**
# - The sentences might not be suitable for this task (it is not word to word translation and you cannot capture the translation)
# - The model might not be powerful enough (Without specifying that this is a translation task, it gives me wrong output as seen in note.ipynb)
# - Attention detector is not accurate
# - threshold might be inappropriate
# - Depending on input, the role of heads could change

# In[46]:
########################################################################
# German

ds_de = load_dataset("Helsinki-NLP/opus-100", "de-en") # German

# %%

de_data = ds_de['train']['translation'][:1000]

de_data = dataset_clean(model, de_data, ('en', 'de'))

# %%

de_sentence_list = loop_dict_to_sentence(de_data, ('en', 'de'), ('EN', 'DE'))

# %%

de_data_rand = randomize_sentences(model, de_data, ("en", "de"))

# %%

de_sentence_rand = loop_dict_to_sentence(de_data_rand, ('en', 'de'), ('EN', 'DE'))

# %%

mean_mat_de = make_mat_mean(model, de_sentence_list, 'DE')

# %%

mean_mat_rand_de = make_mat_mean(model, de_sentence_rand, 'DE')

# In[52]:
########################################################################
# Japanese

ds_ja = load_dataset("Helsinki-NLP/opus-100", "en-ja")

# %%

ja_data = ds_ja['train']['translation'][:1000]

ja_data = dataset_clean(model, ja_data, ('en', 'ja'))

# %%

ja_sentence_list = loop_dict_to_sentence(ja_data, ('en', 'ja'), ('EN', 'JA'))

# %%

ja_data_random = randomize_sentences(model, ja_data, ("en", "ja"))

ja_sentence_rand = loop_dict_to_sentence(ja_data_random, ('en', 'ja'), ('EN', 'JA'))

# %%

mean_mat_ja = make_mat_mean(model, ja_sentence_list, 'JA')

# %%

mean_mat_rand_ja = make_mat_mean(model, ja_sentence_rand, 'JA')

# In[3]:
########################################################################
# Arabic

ds_ar = load_dataset("Helsinki-NLP/opus-100", "ar-en")

# %%

ar_data = ds_ar['train']['translation'][:1000]

ar_data = dataset_clean(model, ar_data, ('en', 'ar'))

# %%

ar_sentence_list = loop_dict_to_sentence(ar_data, ('en', 'ar'), ('EN', 'AR'))

# %%

ar_data_random = randomize_sentences(model, ar_data, ("en", "ar"))

ar_sentence_rand = loop_dict_to_sentence(ar_data_random, ('en', 'ar'), ('EN', 'AR'))

# %%

mean_mat_ar = make_mat_mean(model, ar_sentence_list, 'AR')

# %%

mean_mat_rand_ar = make_mat_mean(model, ar_sentence_rand, 'AR')

# %%
########################################################################
# Spanish

ds_es = load_dataset("Helsinki-NLP/opus-100", "en-es")

# %%

es_data = ds_es['train']['translation'][:1000]

es_data = dataset_clean(model, es_data, ('en', 'es'))

# %%

es_sentence_list = loop_dict_to_sentence(es_data, ('en', 'es'), ('EN', 'ES'))

# %%

es_data_random = randomize_sentences(model, es_data, ("en", "es"))

es_sentence_rand = loop_dict_to_sentence(es_data_random, ('en', 'es'), ('EN', 'ES'))

# %%

mean_mat_es = make_mat_mean(model, es_sentence_list, 'ES')

# %%

mean_mat_rand_es = make_mat_mean(model, es_sentence_rand, 'ES')

# %%

plot_mean_attn(mean_mat_fr, mean_mat_rand_fr)

plot_mean_attn(mean_mat_de, mean_mat_rand_de)

plot_mean_attn(mean_mat_ja, mean_mat_rand_ja)

plot_mean_attn(mean_mat_ar, mean_mat_rand_ar)

plot_mean_attn(mean_mat_es, mean_mat_rand_es)

# %%

# threshold was chosen manually to account for the difference between languages

print(detect_diff_head(mean_mat_fr, mean_mat_rand_fr, 0.075))

print(detect_diff_head(mean_mat_de, mean_mat_rand_de, 0.06))

print(detect_diff_head(mean_mat_ja, mean_mat_rand_ja, 0.04))

print(detect_diff_head(mean_mat_ar, mean_mat_rand_ar, 0.06))

print(detect_diff_head(mean_mat_es, mean_mat_rand_es, 0.06))
