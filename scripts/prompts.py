# prompts.py
"""
Stores prompts, templates, and text-based assets for instruction generation.
"""

import config


# ── Cognitive Levels (Bloom's Taxonomy) ──────────────────────────

QUERY_TYPES = {
    "Remember": "Create instructions that emphasize recall of factual knowledge, definitions, basic concepts, recognition tasks, and core terminology related to the keyword.",
    "Understand": "Design instructions that require conceptual understanding, explanation of relationships, interpretation, illustrative examples, and meaningful comparisons involving the keyword.", 
    "Apply": "Formulate instructions that demand practical use of methods, implementation of procedures, execution of calculations, and real-world application of the keyword.",
    "Analyze": "Develop instructions that involve breaking down complex ideas, identifying patterns, examining relationships, and conducting comparative or structural analysis of the keyword.",
    "Evaluate": "Construct instructions that involve critical judgment, validation of techniques, assessment of alternatives, justification of decisions, and critique of methods related to the keyword.",
    "Create": "Design instructions that foster original thinking, synthesis of ideas, problem innovation, creative design, and novel applications of the keyword."
}


# ── Task Descriptions ────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    "cfa": (
        "Your task is to answer CFA exam questions in a multi-choice form. "
        "Select the correct answer choice (e.g., A, B, C). These questions cover "
        "asset valuation, investment tools, portfolio management, wealth planning, "
        "and ethical and professional standards. They require fundamental knowledge "
        "understanding, quantitative analysis, and application skills."
    ),
    "gsm8k": (
        "You are given a grade school math word problem involving basic arithmetic, "
        "algebra, or geometry. Your task is to carefully read the problem and provide "
        "a step-by-step solution for it."
    ),
    "pubmedqa": (
        "Your task is to answer biomedical research questions based on corresponding "
        "abstracts from PubMed. Determine whether the answer is \"yes,\" \"no,\" or "
        "\"maybe\" by analyzing the provided scientific text. This requires strong "
        "comprehension and interpretation of biomedical literature."
    ),
    "logiqa": (
        "Your task is to solve logical reasoning multiple-choice questions. Analyze "
        "a given logical puzzle or argument and choose the correct option. These "
        "questions test your ability to understand logical structures, identify "
        "fallacies, and make valid inferences."
    ),
    "gpqa": (
        "Your task is to answer challenging, graduate-level multiple-choice questions "
        "spanning Physics, Chemistry, and Biology, requiring deep subject-matter "
        "knowledge, complex reasoning, calculation, and synthesis of information."
    ),
    "math": (
        "You are given a challenging competition math problem. Your task is to "
        "carefully read the problem and provide a step-by-step solution for it."
    ),
    "medqa": (
        "Your task is to answer medical questions collected from professional medical "
        "board exams. These questions test professional-level knowledge across "
        "physiology, pathology, pharmacology, and clinical reasoning. Select the "
        "correct answer choice (e.g., A, B, C, or D)."
    ),
}

# ── Response Suffixes ─────────────────────────────────────────────

_MC_SUFFIX = (
    "Return exactly two lines and nothing else:\n"
    "Reason: <1-3 sentence explanation>\n"
    "Answer: <A|B|C|D>"
)

PROMPT_SUFFIXES = {
    "cfa":      _MC_SUFFIX,
    "logiqa":   _MC_SUFFIX,
    "gpqa":     _MC_SUFFIX,
    "medqa":    _MC_SUFFIX,
    "pubmedqa": (
        "Return exactly two lines and nothing else:\n"
        "Reason: <1-3 sentence explanation>\n"
        "Answer: <yes|no|maybe>"
    ),
    "gsm8k": (
        "Provide a step-by-step reasoning process and then write the final "
        "numerical answer on a new line in the format: final answer: <answer>."
    ),
    "math": (
        "Provide a step-by-step reasoning process and then write the final "
        r"answer in the LaTeX boxed tag: $\boxed{answer}$."
    ),
}


# ── Instruction Generation Templates ─────────────────────────────
# Used by general_instruction_generator.py

SINGLE_KEYWORD_PROMPT_TEMPLATE = """Task Description: {task_description}

Keyword: {keyword}
Cognitive Level: {query_type} - {query_type_description}

Generate a high-quality question that precisely targets the keyword and cognitive level described above. The question should be clear, unambiguous, and suitable for instruction tuning.

Directly output the question. Do not include the answer or any other text.

Generated Question:"""

PAIRED_KEYWORD_PROMPT_TEMPLATE = """Task Description: {task_description}

Keywords: {keyword1}, {keyword2}
Cognitive Level: {query_type} - {query_type_description}

Generate a high-quality, exam-style question that compares, contrasts, or relates the two keywords provided. The question must match the specified cognitive level and be suitable for instruction tuning.

Directly output the question. Do not include the answer or any other text.

Generated Question:"""


# ── Keyword Generation & Bi-Directional Expansion ────────────────
# Used by keywords_generator.py

def _task_desc():
    return TASK_DESCRIPTIONS.get(config.TASK_NAME, "").strip()


def get_initial_keywords_prompt():
    """Prompt for generating the initial seed keyword set."""
    return f"""\
Task Context: You are an expert in {config.DOMAIN} and {config.TASK_NAME}.

Task Description: {_task_desc()}

Instructions: Generate {{count}} core keywords that represent the most \
essential concepts for this task.

Requirements:
- List exactly {{count}} core concepts separated by commas
- Use underscores for multi-word concepts (e.g., {config.EXAMPLE_MULTIWORD})
- Single words are acceptable (e.g., {config.EXAMPLE_SINGLE})
- Provide only the comma-separated list without any other text

Core Keywords:"""


def get_prerequisite_keywords_prompt():
    """Bi-directional expansion — prerequisite (foundational) direction."""
    return f"""\
Task Context: You are an expert in the domain related to: {_task_desc()}

Sample Keywords: {{keywords_str}}

Instructions: What fundamental concepts, basic terminology, or foundational \
principles should learners understand BEFORE studying the sample keywords? \
Generate {{count}} prerequisite concepts.

Requirements:
- List {{count}} prerequisite concepts separated by commas
- Use underscores for multi-word concepts (e.g., {config.EXAMPLE_MULTIWORD})
- Ensure concepts are different from existing keywords
- Provide only the comma-separated list

Prerequisite Concepts:"""


def get_advanced_keywords_prompt():
    """Bi-directional expansion — advanced (specialized) direction."""
    return f"""\
Task Context: You are an expert in the domain related to: {_task_desc()}

Sample Keywords: {{keywords_str}}

Instructions: What specialized subfields, cutting-edge developments, or \
expert-level topics BUILD UPON the sample keywords? Generate {{count}} \
advanced concepts.

Requirements:
- List {{count}} advanced concepts separated by commas
- Use underscores for multi-word concepts (e.g., {config.EXAMPLE_MULTIWORD})
- Ensure concepts are different from existing keywords
- Provide only the comma-separated list

Advanced Concepts:"""


# ── Retrieval-Augmented Keyword Extraction ────────────────────────
# Used by keyword_expansion_from_corpus.py via get_prompt("wikipedia_extraction")

def get_retrieval_augmented_extraction_prompt():
    """Extract keywords from retrieved passages (ArXiv, Wikipedia, etc.)."""
    return f"""\
Task Context: You are an expert in the domain related to: {_task_desc()}

Current Keywords: {{found_keywords}}

Retrieved Passages: The following passages from authoritative sources \
(ArXiv, FreeLaw, StackExchange, Wikipedia, Github) contain comprehensive \
domain knowledge:
{{content}}

The passages already relate to these keywords: {{found_keywords_str}}

Instructions: Extract additional domain-specific keywords directly from \
the retrieved passages that are missing from the current list.

Requirements:
- Use underscores for multi-word concepts (e.g., {config.EXAMPLE_MULTIWORD})
- Single words are acceptable (e.g., {config.EXAMPLE_SINGLE})
- Focus on domain-specific terminology and concepts
- Avoid generic words or concepts already provided
- Provide only the comma-separated list

Extracted Keywords:"""


# ── Prompt Registry ──────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "initial":                get_initial_keywords_prompt,
    "prerequisite":           get_prerequisite_keywords_prompt,
    "advanced":               get_advanced_keywords_prompt,
    "wikipedia_extraction":   get_retrieval_augmented_extraction_prompt,
}


def get_prompt(prompt_type: str):
    """Look up and instantiate a prompt template by name."""
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Prompt type '{prompt_type}' not found. "
            f"Available: {list(PROMPT_TEMPLATES.keys())}"
        )
    return PROMPT_TEMPLATES[prompt_type]()
