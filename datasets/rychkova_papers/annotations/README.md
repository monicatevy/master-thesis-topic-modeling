## Annotations

This folder contains the annotation results produced with **Babelfy**.

### Folder structure
`annotations/`
* `abstract` : only article abstracts are annotated
  * `filtered` : annotations filtered by coherence score
    * `c050` : subfolders are named by threshold (e.g., for coherenceScore ≥ 0.05).
  * `raw` : direct Babelfy output (no coherence filtering)
    * `_meta/` : stores a JSON file used to track documents already processed by Babelfy (to avoid re-sending), and record the accepted chunk size per document
* `fulltext` : annotations on the body text, **excluding** title, abstract, keywords and references.
  * `filtered`
  * `raw`