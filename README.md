# Translation-Heads

## Difference in Attention Head Pattern

Input - An English sentence followed by a sentence in another language. Syntax is as follows.

> EN: This is the largest temple that I've seen. FR: C'est le plus grand temple que j'ai jamais vu.

Used pretrained Qwen2.5-3B and Gemma2-2B.

Computed the mean of the diagonal elements that correspond to the attention from the sentence in another language to the sentence in English.

For the details, please refer to the source code.

The result is as follows. Nonrand means the sentences are translation and rand means it is just a random sentence with similar lengths.

### Qwen2.5-3B

![English to French Qwen](images/EN-FR-qwen.png)

![English to German Qwen](images/EN-DE-qwen.png)

![English to Japanese Qwen](images/EN-JA-qwen.png)

![English to Arabic Qwen](images/EN-AR-qwen.png)

![English to Spanish Qwen](images/EN-ES-qwen.png)

The following is the possible translation heads identified based on the difference in the attention values. I used 0.075 for the threshold except Japanese, whose threshold is 0.05.

FR: ['20.1', '20.4', '20.6', '25.5', '26.12']
DE: ['20.4', '20.6', '25.5', '26.12']
JA: ['20.4', '26.12']
AR: ['20.4', '26.12']
ES: ['15.8', '20.1', '20.4', '20.6', '22.15', '25.5', '26.12']

![French to English Qwen](images/FR-EN-qwen.png)

![German to English Qwen](images/DE-EN-qwen.png)

![Japanese to English Qwen](images/JA-EN-qwen.png)

![Arabic to English Qwen](images/AR-EN-qwen.png)

![Spanish to English Qwen](images/ES-EN-qwen.png)

![Chinese to French Qwen](images/ZH-FR-qwen.png)

![Chinese to German Qwen](images/ZH-DE-qwen.png)

![Chinese to English Qwen](images/ZH-EN-qwen.png)

![French to Chinese Qwen](images/FR-ZH-qwen.png)

![German to Chinese Qwen](images/DE-ZH-qwen.png)

![English to Chinese Qwen](images/EN-ZH-qwen.png)


### Gemma2-2B

[French Gemma]
*English-French translation in Gemma*

[German Gemma]
*English-German translation in Gemma*

[Japanese Gemma]
*English-Japanese translation in Gemma*

[Arabic Gemma]
*English-Arabic translation in Gemma*

[Spanish Gemma]
*English-Spanish translation in Gemma*

The following is the possible translation heads identified based on the difference in the attention values. I used 0.04 for the threshold except Japanese, whose threshold is 0.03.

FR: ['6.2', '8.1', '10.4', '15.0']
DE: ['5.2', '6.1', '6.2', '8.1', '8.6', '10.4', '10.5', '15.0']
JA: ['6.2', '8.1', '8.6', '10.4', '12.3', '15.0', '16.4', '16.7']
AR: ['6.2', '8.1', '8.6', '10.4', '10.5', '15.0']
ES: ['6.2', '8.1', '10.5', '15.0']
