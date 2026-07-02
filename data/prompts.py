"""Instruction templates for QwenVLSeg — ChatML format with bbox + mask tokens.

Current project uses Qwen3-VL-2B-Instruct (no thinking mechanism).
Prompt format: system + user (with vision tokens) + assistant (with JSON target).
"""

# Special tokens for mask decoding
MASK_START = "<mask_start>"
MASK_TOKEN = "<mask_token>"
MASK_END = "<mask_end>"


def build_multi_category_prompt(categories: list) -> str:
    """Build user instruction for multi-category referring segmentation."""
    categories_str = ", ".join(categories)
    return (
        f"Locate and segment every instance that belongs to the following categories "
        f'"{categories_str}", report bbox coordinates and masks in JSON format.'
    )


def build_category_prompt(category: str) -> str:
    """Build user instruction for single-category referring segmentation."""
    return build_multi_category_prompt([category])


def build_target_json(bbox_list: list, category: str) -> str:
    """Build target JSON with bbox coordinates + mask token placeholders.

    Args:
        bbox_list: list of [x1, y1, x2, y2] in 0-1000 coords
        category: category name

    Returns:
        Markdown code block containing JSON array with bbox + mask tokens
    """
    items = []
    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox
        items.append(
            "\t{"
            f'"bbox_2d": [{x1}, {y1}, {x2}, {y2}], '
            f'"label": "{category}", '
            f'"mask": "{MASK_START}{MASK_TOKEN}{MASK_END}"'
            "}"
        )
    json_content = "[\n" + ",\n".join(items) + "\n]"
    return f"\n```json\n{json_content}\n```"



