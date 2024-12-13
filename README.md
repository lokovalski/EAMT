# How to make sense of the junk drawer this currently is this repo.

*Original Data:*
- data/spanish.jsonl
- data/italian.jsonl

**Data Processing/SemEval Data Quality Analysis**
- data_processing.py (ran to fix errors in json)
- ner_processing.py (used in determining whether to use sem eval labels or SpanMarker predictions as source of entity truth)

*Final Cleaned Data* 
- data/spanish_train.json, spanish_test.json, spanish_dev.json
- data/italian_train.json, italian_test.json, italian_dev.json

**Pipeline MT Code (used to run our proposed model on the final cleaned data)**
- Lola-Spanish-Test.ipynb
- Ina-Italian-Test.ipynb

*Test Output from Pipelined EAMT Model:*
- data/translated_spanish_test_new.json
- data/translated_italian_test_new.json

**Meta M2M Translation Code:**
- Lola-Spanish-Test-MetaModel.ipynb
- Ina-Italian-Test-MetaModel.ipynb

*Test Output from Meta Model:*
- meta_translation_italian.json
- meta_translation_spanish.json

Figures and metrics for the paper were created by running various scripts in lolatest.ipynb.
