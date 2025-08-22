# Master Thesis — Topic Modeling (CREA vs LLMs)
This repo contains the code and experiments for my master’s thesis:
**Comparing CREA and Prompt-based LLMs for Topic Modeling**.

## Thesis scope
- Follows **Fabrice Boissier’s CREA thesis** (FCA-based pipeline) as the starting point.
- Addresses a research gap: there are few empirical studies using a FCA-based method for topic modeling.
- Compares a transparent CREA pipeline with LLM-based (black-box) pipeline

## Pipelines
- **CREA (FCA + clustering):**
  1) Preprocess (cleaning, TreeTagger lemmatization)
  2) Babelfy entity linking → annotations CSV → filter by coherence
  3) Occurrence matrices → normalization → binarization
  4) FCA context export (.cxt)
  5) HCA Clustering
- **Prompt-based LLMs:**
  1) Topic Generation
  2) Topic Labeling

## Datasets
- PHP Courses
- 20 Newsgroups
- Rychkova Research Papers (abstracts & full text)
