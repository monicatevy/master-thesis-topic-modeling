# CREA Pipeline (Draft)

## Purpose
CREA produces interpretable topics by building a **document–entity context**, applying **Formal Concept Analysis (FCA)**, then **clustering** the resulting concepts into topics.

## Steps
0. **Inputs**  
   Text sourced from the selected datasets.

1. **Preprocessing**  
   Clean text (line breaks, noise), remove stopwords, lemmatize, and apply POS/class filters using **TreeTagger**.

2. **Entity Annotation**  
   Submit the (raw or preprocessed) texts to **Babelfy** for entity linking.

3. **Filtering by Coherence Score**  
   Keep only annotations with **coherence ≥ 0.05** (current default threshold).

4. **Representation Construction**  
   Build the **document–entity context** (binary or weighted).  
   Optionally compute **entity co-occurrence** for auxiliary analyses.

5. **FCA**  
   Construct the **formal context** and compute the **concept lattice**.  
   Use concept **intents/extents** as semantic units.

6. **Clustering & Topics**  
   Cluster concepts to group related ones and form topics.  
   Derive concise topic descriptors (top entities/terms per cluster), pruning generic terms when needed.

7. **Metrics (separate module/folder)**  
   Compute intrinsic topic quality metrics (**C_v**, **C_npmi**, **U_Mass**, **U_CI**, **Topic Uniqueness**).
