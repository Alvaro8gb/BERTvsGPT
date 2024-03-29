# Spanish EHR Structuring Comparative Analysis BERT vs. GPT

## Abstract

**Purpose:**
In the era of healthcare data digitization, effective methods for structuring electronic health records (EHRs) are crucial. This study conducts a comparative analysis between the traditional Named Entity Recognition (NER) method using BERT and a contemporary Large Language Model (LLM)-driven approach using GPT, focusing on structuring Spanish EHRs. The research assesses the effectiveness, accuracy, and applicability of both methodologies.

**Methods:**
This study utilizes a dataset of Spanish EHRs related to breast cancer. It implements a traditional NER method using BERT and a contemporary approach combining few-shot learning and external knowledge integration, driven by Large Language Models (LLMs) using GPT. The analysis involves a comprehensive pipeline, and key performance metrics (precision, recall, F1 score) are employed for evaluation. The goal is to highlight the strengths and limitations of each method in structuring Spanish EHRs.

**Results:**
The comparative analysis demonstrates that both the traditional BERT-based NER method and the few-shot LLM-driven approach, augmented with external knowledge, provide comparable accuracy levels in metrics such as precision, recall, and F1 score for Spanish EHRs. Contrary to expectations, the LLM-driven approach, requiring minimal data annotation, performs on par with BERT in discerning complex medical terminologies and contextual nuances.

**Conclusion:**
This study marks a significant advancement in Spanish EHR Named Entity Recognition. The few-shot LLM-driven approach, enhanced by external knowledge, slightly outperforms the traditional BERT-based method in overall effectiveness. GPT's superiority in F-score and minimal reliance on extensive data annotation highlight its potential in medical data processing.

---

## Code Repository

### Repository Structure

This repository is organized into the following sections:

#### `BERT` [ref](./BERT)

This section contains the code for performing Named Entity Recognition (NER) using BERT. The implementation includes a K-fold cross-validation approach for training. Here is an example of an inference from a clinical note using the BERT model:

   <img src="./BERT/ner-anotation.png" width="800" alt="Zero Prompt">


#### `GPT` [ref](./GPT)

In this section, you'll find the code for performing Named Entity Recognition (NER) with GPT. 
The use the  `@@entity##` format to extract entities from the clinical notes. 

- Zero-shot Prompt Sample:

   <img src="./GPT/promts/zero-promt.png" width="700" alt="Zero Prompt">


- Few-shot Prompt Sample:

   <img src="./GPT/promts/few-promt.png" width="700" alt="Few Prompt">

#### `data` [ref](./data)

The `data/` directory does not contain the actual datasets due to privacy agreements with the hospital. However, scripts for conducting Exploratory Data Analysis (EDA) and guidelines on how to load the corpus are provided. These resources offer valuable insights into the dataset structure and loading procedures while respecting privacy constraints.

---

### Help 
Feel free to explore each section for detailed implementations and insights into the respective NER methods. If you have any questions or concerns, please refer to the documentation or contact the repository maintainers.

---

### Set up

Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Acknowledgments

- This research is based on the paper [Title of the Paper], published in [Journal Name].
- Authors: [Author1], [Author2], ...



