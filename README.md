# Toxicity Detector: a Repository

This research explores increasing the efficiency and accuracy of LLM-based toxic content detection.

## Toxic Content Detector: Using Retrieval-Augmented Generation and Batch Processing to Improve Accuracy and Efficiency of LLM-based Toxic Content Detection

### Project Overview

As the influence of social media in the 21st century continues to rise, a growing concern that rises with it is toxic content: hate speech, misinformation, and discrimination. Previous research identifies the efficiency benefits of using Large Language Models (LLMs) to detect toxic content, but current approaches still result in classification errors and can have a lack of adaptability in the approach. This project presents a novel method that combines retrieval-augmented generation (RAG) with batch processing to improve the accuracy and efficiency of toxic content detection using LLMs. By supplementing the LLM's internal knowledge with relevant data from an external database, RAG enhances the LLM's decision-making when classifying toxic content. Simultaneously, batch processing reduces the processing time per sentence by not re-prompting the LLM for every individual sentence. We evaluate the proposed method using [GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence) and [ToxiGen](https://arxiv.org/abs/2203.09509) dataset. Our experimental results demonstrate that processing sentences in batches of 10 reduces the response time per sentence by 49.5%, and RAG increases toxicity detection accuracy by 7% for a batch size of 10. With increased adaptability, accuracy, and efficiency, the proposed approach is a practical and scalable method to boost LLM detection of toxic content in digital media.
This research project was submitted to the 2025 Conference on Machine Learning and Automation London Symposium for publication.

## Cloning This Repository

Clone this repo with

```
git clone git@github.com:saphiooo/toxicity_detector.git
```

## Replicating This Repository

Install the necessary libraries:

```pip install openai
pip install pandas
pip install torch
pip install scikit-learn
pip install time
pip install re
```

Enter the repository with

```
cd toxicity-detector
```

and run the main Python file with

```
python3 main.py
```

Toxicity Detector uses ToxiGen training and testing data, and is currently set to parse data organized in that format. More general adaptability will be added in future versions.

**NOTE**: In `main.py`, your API key must replace the placeholder in line 28.

## Known Issues/Future Improvement

Knwon areas of improvement include:

-   Handling links, images, and embedded content in input
-   Expansion to accomodate other dataset layouts
