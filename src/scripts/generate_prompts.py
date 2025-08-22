import os
import warnings
from pathlib import Path

from src.llm.prompt.utils import build_prompts, export_prompt_config

warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from src.llm.prompt.templates import build_prompt_parts

start = time.time()

DATASET_PATH = "/Users/monicasen/PycharmProjects/topic-modeling/datasets/rychkova_papers"
INPUT_CSV = os.path.join(DATASET_PATH, "data", "abstracts.csv")
PROMPT_DIR = os.path.join(DATASET_PATH, "prompts")

DATA_NAME = Path(INPUT_CSV).stem
VARIANT_NAME = "natural_2"
FILENAME_BASE = f"{DATA_NAME}_{VARIANT_NAME}"

DOC_IDS = {"2025-LR", "2025-AR"}
HEADER, FOOTER = build_prompt_parts(VARIANT_NAME)

build_prompts(
    input_path=INPUT_CSV,
    output_dir=PROMPT_DIR,
    prompt_header=HEADER,
    prompt_footer=FOOTER,
    source_type="csv",
    # filter_ids=DOC_IDS
    filename_prefix=FILENAME_BASE
)

timestamp = time.strftime("%Hh%M")
OUTPUT_ZIP_DIR = os.path.join(PROMPT_DIR, "zip_exports")
ZIP_NAME = f"{DATA_NAME}_{VARIANT_NAME}_{timestamp}"

export_prompt_config(
    prompt_dir=PROMPT_DIR,
    output_zip_dir=OUTPUT_ZIP_DIR,
    zip_name=ZIP_NAME
)

end = time.time()
duration = end - start
print(f"⏱️ Done in {duration:.2f} sec ({duration/60:.2f} min).")