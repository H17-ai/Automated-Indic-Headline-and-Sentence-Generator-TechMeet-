# Automated-Indic-Headline-and-Sentence-Generator-TechMeet-
Automated identification, summarization, and entity-based sentiment analysis of mobile technology articles and tweets.

**Problem:**
*  Create a smart system that can recognise the themes in tweets and articles. The sentiments against a brand (at the tweet/paragraph level) should be identified if the theme is mobile technology.
*  For articles that adhere to the mobile technology subject, we would need a one-sentence headline of 20 words.  A headline for tweets is not required.

## 0. Data-Description
For this task, we received a mix of 4000 non-mobile tech and mobile tech tweets and articles each, with their labels of mobile_tech_tag as well as the headlines for the articles. No data was given for entity-based sentiment analysis. 

## 1. Pipeline
![Pipeline](https://github.com/Raghu-vamsi-pav/Automated-Indic-Headline-and-Sentence-Generator-TechMeet-/blob/main/assets/pipeline.png)
We developed an end-to-end pipeline that runs from the input to the output while assuring good efficiency and code scalability.

### 1.1  Preprocessing

We created a preprocessing pipeline to remove all the useless portions of the text and leave the valuable features. For that, we did the following:

*  Remove the Hyperlinks and URLs
*  Segment the hashtags
*  Demojify the emojis
*  Remove punctuations
*  Convert the text into lower-case

The majority of tweets and articles were repeated numerous times after our dataset had gone through the preprocessing method.

Since many examples were distinct due to the presence of some gibberish or extremely uncommon terms, we devised a special graph-based clustering technique employing Levenshtein distance for the removal of duplicate sentences. We were unable to locate them using only the duplicate techniques already present in pandas. Therefore, we created a clustering algorithm that groups related phrases with a Lavestein distance below a predetermined threshold, and from each cluster, a single data point is chosen as an original example.

Only 10% of the sample had unique text after we eliminated the duplicates, leaving about 400 tweets and articles.

### 1.2 Language Detection

We created a Bidirectional LSTM-based model for the language detection module that was trained on a custom dataset of Hinglish and English words. The model developed a 93% accuracy rate for recognising words in Hinglish.

We can separate tweets and articles into code-mixed, pure-Indic language, and English sentences by using language detection to determine the language of the tweet or article.

### 1.3  Transliteration and Translation
Dealing with code-mixed languages in our dataset was one of the main hurdles. Languages that are code-mixed are rather prevalent in nations with bilingual and multilingual cultures. We considered transliterating the phrases to their intended languages in order to cope with code-mixed language and transform it into a pure-Indic language.

We first had to gather the necessary data in order to build a transliteration model. The following sites served as our main data sources:

* Xlit-Crowd: Hindi-English Transliteration Corpus
These pairs were obtained via crowdsourcing by asking workers to
transliterate Hindi words into the Roman script. The tasks were done on
Amazon Mechanical Turk and yielded a total of 14919 pairs.
* NeuralCharTransliteration: Data is created from the scrapped Hindi songs lyrics.
* Xlit-IITB-Par: This is a corpus containing transliteration pairs for Hindi-English. These pairs were automatically mined from the IIT Bombay English-Hindi Parallel Corpususing the Moses Transliteration Module. The corpus contains 68,922 pairs.
* BrahmiNet Corpus: 110 language pairs
* Hindi word transliteration pairs
* Dakshina dataset: The Dakshina dataset is a collection of text in both Latin and native scripts for 12 South Asian languages. For each language, the dataset includes a large collection of native script Wikipedia text, a romanization lexicon which consists of words in the native script with attested romanizations, and some full sentence parallel data in both a native script of the language and the basic Latin alphabet.

The aforementioned datasets were combined, cleaned, and turned into a final data file for our model to be trained (available in the data folder). Once the data file was created, we trained a transformer model on it because this work required speed and transformers are faster than RNNs due to parallelism.

### 1.4  Classification of mobile_tech text

We utilised a stacked BiLSTM on the preprocessed text to categorise the input data into tech and non-tech. Because the BiLSTM model produced results similar to those of the transformer-based model while also being faster, it was chosen over the latter.

We concatenated 80% of the given data with the scraped dataset, and we utilised this combined data for training. This helped the model understand the given data distribution and generalise to newer out-of-distribution samples as well.

### 1.5 Sentiment analysis based on brand identification and aspect
We are employing a dictionary-based strategy to identify brands in mobile tech tweets and articles, and the accuracy can be further improved by using a transformer for NER (Named Entity Recognition).

A transformer-based model is used for feature extraction in aspect-based sentiment analysis, which is then fed into a GRU model to determine the sentiment of each word in the phrase. Finally, we will choose the brands' sentiments at random.

Attention mechanisms and Convolutional Neural Networks (CNNs) are frequently used for aspect-based sentiment classification due to their intrinsic capacity for semantic alignment of aspects and the words that describe them. These models might falsely identify syntactically irrelevant surrounding words as cues for determining aspect sentiment, however, because they lack a way to account for relevant syntactical restrictions and long-range word dependencies. Utilizing word dependencies and syntactical information, we are utilising a [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1909.03477) over the dependency tree of a phrase to address this issue. On the basis of it, a framework for aspect-specific sentiment classification is developed.

### 1.6 Heading generation
This task involves passing a given article or document via a network to produce a summary. There are two kind of mechanisms for producing summaries:

1. ***Extractive Summary:*** To give the most insightful information from the article, the network compiles the most significant sentences from the text.
2. ***Abstract Summary***: The network generates new sentences that capture the essence of the content as fully as possible. The article may or may not contain the sentences from the synopsis.

In this section, we will create a ***Abstract Summary***.

 **Data**:
 
 - We are using the [Kaggle](https://www.kaggle.com/sunnysai12345/news-summary) News Summary dataset. This dataset is a compilation of Indian-published newspapers.
 - Additionally, we scraped headlines from stories about mobile technology on https://gadgets.ndtv.com/mobiles.



**Language Model Used**: 
 - One of the newest and most innovative transformer models, the ***T5***, is used in this notebook. (Research Report) (https://arxiv.org/abs/1910.10683)
 - ***T5*** is one of a kind in many aspects, with a transformers architecture that not only does several NLP jobs at the cutting edge of technology, but also takes a highly unconventional approach to them.
 - **Text-2-Text** - based on the illustration from the T5 paper. Every NLP task is transformed into a **text-to-text** issue. Instead of being viewed as independent, distinct problem statements, tasks like translation, classification, summarization, and question answering are all regarded as a text-to-text conversion challenge.
     
## 2. Future Prospects
- The translation part of the pipeline can be scaled to multiple code-mixed language.
We can create a model to automatically learn the reverse mapping between brand and model with the availability of more data since it was stated that when it comes to different models of a brand, we need to identify them only if they appear with their brand name. For example, galaxy needs to be identified as a brand only if it is mentioned in the text with Samsung.
- With the availability of more data, the pipeline can be expanded to include other industries, such as fashion and transportation.
- It is possible to utilise the entire pipeline for business purposes.

## 3. Acknowledgments and References
- We have utilised several open source libraries, datasets, and available codebase for various pipeline components. We appreciate the creators of the aforementioned libraries, codebases, and datasets.

- [ASGCN Codebase](https://github.com/GeneZC/ASGCN)
- [ASGCN Paper](https://arxiv.org/abs/1909.03477)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [T5 Hugging face transformer](https://huggingface.co/transformers/model_doc/t5.html)
- [News Summary Dataset Kaggle](https://www.kaggle.com/sunnysai12345/news-summary)
- [MarianMT](https://huggingface.co/transformers/model_doc/marian.html)
- [Pytorch](https://pytorch.org/)
