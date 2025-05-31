from jaxtyping import Float, Int
import matplotlib.pyplot as plt
import torch as t
from torch import Tensor
from transformer_lens import (
    ActivationCache,
    HookedTransformer,
    utils,
)

def dataset_clean(model: HookedTransformer,
        ds: list[dict], lang: tuple[str], len_diff: int = 5, 
        min_len:int = 3, max_len:int = 50) -> list[dict]:
    # returns clean sentences from the dataset
    # lang has to be a language that has space inbetween the words
    en = lang[0]
    other = lang[1]
    new_dict = []
    for data in ds:
        sentence_en = data[en]
        sentence_other = data[other]
        if (len(sentence_en.split()) < min_len or 
            len(sentence_en.split()) > max_len):
            continue
        elif abs(len(model.to_str_tokens(sentence_en)) - 
                 len(model.to_str_tokens(sentence_other))) > len_diff:
            continue
        else:
            sentence = sentence_en + sentence_other
            if any(char.isdigit() for char in sentence):
                continue
            elif any(char == '%' for char in sentence):
                continue
            elif sentence.startswith('-') or sentence_other.startswith('-'):
                continue
            elif any(char == '-' for char in sentence):
                continue
            elif any(char == '_' for char in sentence):
                continue
            elif any(char == '\\' for char in sentence):
                continue
            elif any(char == '/' for char in sentence):
                continue
            elif any(char == '*' for char in sentence):
                continue
            elif any(char == '&' for char in sentence):
                continue
            elif any(char == '\u3000' for char in sentence):
                continue
            elif any(char == '♪' for char in sentence):
                continue
            else:
                new_dict.append(data)
    return new_dict

def detect_diff_head(tensor1: Tensor, tensor2: Tensor, threshold: float) -> list[str]:
    diff = tensor1 - tensor2
    x, y = diff.shape
    diff_heads = []
    for i in range(x):
        for j in range(y):
            if diff[i][j] > threshold:
                diff_heads.append(str(i) + "." + str(j))
    return diff_heads

def find_translation_token(
        model: HookedTransformer, tokens, language_heads: list[str]) -> list[int]:
    str_tokens = model.to_str_tokens(tokens)
    head_index = []
    for head in language_heads:
        # if 2, that means it is not the language token
        head_index.append(str_tokens.index(head, 3))
    return head_index

def loop_dict_to_sentence(
        data: list[dict], lang_lower: tuple[str], lang_upper: tuple[str]) -> list[str]:
    anchor = lang_lower[0]
    translated = lang_lower[1]
    ANCHOR = lang_upper[0]
    TRANSLATED = lang_upper[1]
    sentences = []
    for element in data:
        sentences.append(f"{ANCHOR}: {element[anchor]} {TRANSLATED}: {element[translated]}")
    return sentences

def make_mat_mean(model: HookedTransformer, sentences: list[str], lang: str) -> Float[Tensor, "n_layers n_heads"]:
    mean_mat = t.zeros([model.cfg.n_layers, model.cfg.n_heads])
    for sentence in sentences:
        tokens = model.to_tokens(sentence)
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        token_lang = find_translation_token(model, tokens, [f" {lang}"])
        mat = translation_attn_detector_mat(model, cache, 1, token_lang[0])
        mean_mat += mat
    mean_mat /= len(sentences)
    return mean_mat

def plot_mean_attn(tensor1, tensor2, lang):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    vmin = min(tensor1.min(), tensor2.min())
    vmax = max(tensor1.max(), tensor2.max())

    # Visualize tensor1
    im1 = axs[0].imshow(tensor1, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('Translation')
    axs[0].set_xlabel('Head')
    axs[0].set_ylabel('Layer')
    fig.colorbar(im1, ax=axs[0])

    # Visualize tensor2
    im2 = axs[1].imshow(tensor2, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('Random')
    axs[1].set_xlabel('Head')
    axs[1].set_ylabel('Layer')
    fig.colorbar(im2, ax=axs[1])

    # Visualize the difference
    im3 = axs[2].imshow(tensor1 - tensor2, cmap='viridis')
    axs[2].set_title('Diff')
    axs[2].set_xlabel('Head')
    axs[2].set_ylabel('Layer')
    fig.colorbar(im3, ax=axs[2])


    plt.subplots_adjust(bottom=0.1, right=1.1)
    fig.suptitle(f"Cross language attention from {lang[0]} to {lang[1]}", fontsize=16)

    plt.show()

def randomize_sentences(
        model: HookedTransformer, ds: list[dict], lang: tuple[str]) -> list[dict]:
    temp = []
    en = lang[0]
    other = lang[1]
    for data in ds:
        n = len(model.to_str_tokens(data[other]))
        temp.append((n, data))
    temp = sorted(temp, key=lambda x: x[0])
    random_dict = []
    if len(ds) % 2 == 1:
        first = temp.pop()[1]
        second = temp.pop()[1]
        third = temp.pop()[1]
        dict1 = {en: first[en], other: second[other]}
        dict2 = {en: second[en], other: third[other]}
        dict3 = {en: third[en], other: first[other]}
        random_dict.extend([dict1, dict2, dict3])
    while (len(temp) > 0):
        first = temp.pop()[1]
        second = temp.pop()[1]
        dict1 = {en: first[en], other: second[other]}
        dict2 = {en: second[en], other: first[other]}
        random_dict.extend([dict1, dict2])
    return random_dict

def translation_attn_detector(
        model: HookedTransformer, cache: ActivationCache, bandwidth: int, 
        threshold: float, token_idx: list[int], bos: int) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4"] of "layer.head" which you judge to be translation heads
    bandwidth is how much offset you allow the diagonal to be and bandwidth = 0 means only look at the diagonal itself
    token_idx is the idx of a sentence in a new languages
    bos is the number of bos tokens at the start of the sentence
    """
    translation_head = []
    for i in range(model.cfg.n_layers):
        attn_pattern = cache["pattern", i]
        for j in range(model.cfg.n_heads):
            mat = attn_pattern[j]
            d_mean = 0
            for idx in token_idx:
                diag_mean = 0
                diag_mean += mat.diagonal(-idx + 1 + bos).mean()
                for k in range(1, bandwidth + 1):
                    diag_mean += mat.diagonal(- idx + 1 + k + bos).mean()
                    diag_mean += mat.diagonal(- idx + 1 - k + bos).mean()
                d_mean += diag_mean / (1 + 2 * bandwidth)
            if d_mean / len(token_idx) > threshold:
                translation_head.append("" + str(i) + "." + str(j))
    return translation_head


def translation_attn_detector_normalized(
        model: HookedTransformer, cache: ActivationCache, bandwidth: int, 
        threshold: float, token_idx: list[int], bos: int=1, lang_head: bool=False) -> list[str]:
    """
    translation_attn_detector but normalize with the value inside the square
    Only compare the pair of the first language and the language in target
    lang_head = True means to account for the phenomeno where the attention concentrates to "EN"
    """
    # assuming that there is no space after the last token
    translation_head = []
    n_heads, x, y = cache["pattern", 0].shape
    n_language = len(token_idx)
    token_idx.append(y)
    alpha = 0
    if lang_head:
        alpha = 1
    for i in range(model.cfg.n_layers):
        attn_pattern = cache["pattern", i]
        for j in range(model.cfg.n_heads):
            mat = attn_pattern[j]
            epsilon = attn_pattern[:, bos+alpha:].mean()
            epsilon_head = attn_pattern[:, :bos+alpha].mean()
            mean_ratio = 0
            for k in range(n_language):
                cropped_mat = mat[token_idx[k]+alpha:token_idx[k+1], bos+alpha:token_idx[0]]
                diag_mean = 0
                diag_mean += cropped_mat.diagonal(-1).mean()
                for l in range(1, bandwidth + 1):
                    diag_mean += cropped_mat.diagonal(l-1).mean()
                    diag_mean += cropped_mat.diagonal(-l-1).mean()
                diag_mean /= (1 + 2*bandwidth) # mean activation value across diagonal
                cropped_mean = cropped_mat.mean()
                mean_ratio += diag_mean / (cropped_mean + 0.75 * epsilon + 0.1 * epsilon_head)
            if mean_ratio / n_language > threshold:
                translation_head.append(str(i) + "." + str(j))
    return translation_head

def translation_attn_detector_mat(
        model: HookedTransformer, cache: ActivationCache, bandwidth: int, 
        token_idx: int, bos: int=1) -> Float[Tensor, "n_layers n_heads"]:
    """
    translation_attn_detector but normalize with the value inside the square
    Only compare the pair of the first language and the language in target
    lang_head = True means to account for the phenomeno where the attention concentrates to "EN"
    return a matrix
    """
    # assuming that there is no space after the last token
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    diag_mat = t.zeros([n_layers, n_heads])
    for i in range(n_layers):
        attn_pattern = cache["pattern", i]
        for j in range(n_heads):
            mat = attn_pattern[j]
            cropped_mat = mat[token_idx:, bos:token_idx]
            diag_mean = 0
            diag_mean += cropped_mat.diagonal().mean()
            for l in range(1, bandwidth + 1):
                diag_mean += cropped_mat.diagonal(l).mean()
                diag_mean += cropped_mat.diagonal(-l).mean()
            diag_mean /= (1 + 2*bandwidth) # mean activation value across diagonal
            diag_mat[i][j] = diag_mean
    return diag_mat
