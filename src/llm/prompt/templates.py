# prompt_templates.py

# ───── SYSTEM PROMPTS ─────
PROMPT_SYSTEM_1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant specialized in topic modeling. Always respond in valid JSON format only.
<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

PROMPT_SYSTEM_2 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert language model trained to extract meaningful topics from raw text.
You must follow the instructions carefully and only produce valid XML — no explanations, no extra text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

PROMPT_SYSTEM_3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert language model trained to extract meaningful topics from raw text.
You must follow the instructions carefully — no explanations, no extra text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

# ───── INSTRUCTION PROMPTS ─────
NUM_TOPICS = 8
NUM_KEYWORDS = 8

BASE_TM_DIRECT = f"""Write the results of simulating topic modeling for the following documents.
Assume you will identify {NUM_TOPICS} topics, each composed of {NUM_KEYWORDS} representative words."""

TM_ORIGINAL = """Write the results of simulating topic modeling for the following documents, each starting with "#." NOTE: Outputs must always be in the format "Topic k: word word word word word" and nothing else."""

TM_SIMPLE = """Write the results of simulating topic modeling for the following documents."""

TM_SOFT_CONSTRAINTS = """Perform topic modeling on the following documents. Identify a set of coherent topics. For each topic, return a few representative keywords that capture its meaning."""

TM_WITH_NB_WORDS = """Perform topic modeling on the following documents. Identify a set of coherent topics based on the content. For each topic, return 4 to 8 representative keywords that best capture its meaning."""

TM_EXPLICIT = """Perform topic modeling on the following documents. Outputs must always be in the format "Topic k: word word word word word". Do NOT include any titles, labels, or descriptions. Do NOT explain the topics."""

# ───── OUTPUT FORMAT PROMPTS ─────
OUTPUT_NATURAL = """Return output like: Topic k : word word word word
"""

OUTPUT_JSON = """Return ONLY valid JSON output like:
[
  { "id": "topic_1", "keywords": ["...", "...", "..."] }
]
"""

OUTPUT_XML = """Return ONLY valid XML output like:
  <topics>
    <topic>
      <word></word>
      <word></word>
      <word></word>
      <word></word>
      <word></word>
    </topic>
  </topics>
"""

# ───── CONFIG DICTIONARY ─────
PROMPT_VARIANTS = {
    "natural_1": {
        "system": PROMPT_SYSTEM_3,
        "instruction": TM_SIMPLE,
        "format": OUTPUT_NATURAL,
    },
    "natural_2": {
        "system": PROMPT_SYSTEM_3,
        "instruction": TM_ORIGINAL
    },
    "natural_3": {
        "system": PROMPT_SYSTEM_3,
        "instruction": TM_SOFT_CONSTRAINTS,
        "format": OUTPUT_NATURAL,
    },
    "natural_4": {
        "system": PROMPT_SYSTEM_3,
        "instruction": TM_WITH_NB_WORDS
    },
    "natural_5": {
        "system": PROMPT_SYSTEM_3,
        "instruction": TM_EXPLICIT
    }
}

# ───── HELPER ─────
def build_prompt_parts(variant_name: str) -> tuple[str, str]:
    cfg = PROMPT_VARIANTS[variant_name]
    prompt_format = cfg.get('format', '')

    header = f"""{cfg['system']}<task>
  <instruction>
    {cfg['instruction']}{prompt_format}</instruction>
  <documents>
"""
    footer = """</documents>
</task>
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return header, footer
