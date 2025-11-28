"""
Prompt management module for image captioning.

Contains all prompt templates and prompt-related functions for the caption system.
"""

from typing import Dict, Any, List, Tuple

# Built-in prompt templates
CAPTION_PROMPTS = {
    'SIMPLE': "Describe this image in detail.",
    
    'SHORT': "Write a very short caption for the given image.",
      
    'LONG': "Write a very long and detailed caption describing the given image as comprehensively as possible."
}


def get_prompt(prompt_name: str) -> str:
    """Get a specific prompt from the built-in prompt dictionaries."""
    # First check in caption prompts
    if prompt_name in CAPTION_PROMPTS:
        return CAPTION_PROMPTS[prompt_name]
    # Fallback to simple prompt
    return CAPTION_PROMPTS['SIMPLE']

def create_taxonomy_prompts(taxonomy: Dict[str, Any], prompt_name: str = "default") -> str:
    """Create a comprehensive prompt from taxonomy structure.
    
    Args:
        taxonomy: The taxonomy dictionary structure
        prompt_name: Type of prompt to generate ('default' or 'structured')
    
    Returns:
        A formatted prompt string
    """
    if prompt_name == "structured":
        print("Creating structured taxonomy prompt")
        return _create_structured_taxonomy_prompt(taxonomy)
    else:
        print("Creating default taxonomy prompt")
        return _create_default_taxonomy_prompt(taxonomy)


def _create_default_taxonomy_prompt(taxonomy: Dict[str, Any]) -> str:
    """Create a default list-based taxonomy prompt."""
    prompt_lines = ["Describe this image from the following perspectives. Skip if no information applies to the specific aspect.\n"]
    
    for big_cat, sub_cats in taxonomy.items():
        if isinstance(sub_cats, dict):
            for sub_cat, items in sub_cats.items():
                if isinstance(items, dict):
                    # Nested dictionary - handle third level
                    for item, details in items.items():
                        if isinstance(details, list) and details:
                            examples = ', '.join(str(d) for d in details)
                            prompt_lines.append(f"- {big_cat} -> {sub_cat} -> {item} (e.g., {examples})")
                        else:
                            prompt_lines.append(f"- {big_cat} -> {sub_cat} -> {item}")
                elif isinstance(items, list) and items:
                    # Has examples
                    examples = ', '.join(str(item) for item in items)
                    prompt_lines.append(f"- {big_cat} -> {sub_cat} (e.g., {examples})")
                else:
                    # No examples
                    prompt_lines.append(f"- {big_cat} -> {sub_cat}")
        else:
            prompt_lines.append(f"- {big_cat}")
    
    return '\n'.join(prompt_lines)


def _create_structured_taxonomy_prompt(taxonomy: Dict[str, Any]) -> str:
    """Create a structured JSON-based taxonomy prompt.
    
    Note: For forced structured output, use create_taxonomy_json_schema() instead.
    """
    import json
    
    # Build the JSON template structure
    json_template = {}
    for big_cat, sub_cats in taxonomy.items():
        if isinstance(sub_cats, dict):
            json_template[big_cat] = {}
            for sub_cat, items in sub_cats.items():
                if isinstance(items, dict):
                    # Nested dictionary - handle third level
                    json_template[big_cat][sub_cat] = {}
                    for item, details in items.items():
                        if isinstance(details, list) and details:
                            examples = ', '.join(str(d) for d in details)
                            json_template[big_cat][sub_cat][f"{item} (e.g., {examples})"] = ""
                        else:
                            json_template[big_cat][sub_cat][item] = ""
                elif isinstance(items, list) and items:
                    # Has examples
                    examples = ', '.join(str(item) for item in items)
                    json_template[big_cat][f"{sub_cat} (e.g., {examples})"] = ""
                else:
                    # No examples
                    json_template[big_cat][sub_cat] = ""
        else:
            json_template[big_cat] = ""
    
    # Format the JSON template
    json_str = json.dumps(json_template, indent=2)
    
    prompt = f"""Describe this image by filling in the following JSON structure. Write None if no information applies to the specific aspect.

{json_str}

Return the completed JSON with descriptions filled in for each applicable field."""
    
    return prompt


def create_taxonomy_json_schema(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """Create a JSON schema for structured output based on taxonomy.
    
    This schema can be used with structured output APIs like:
    - OpenAI's response_format
    - vLLM's guided_json
    - Gemini's response_schema
    
    Args:
        taxonomy: The taxonomy dictionary structure
    
    Returns:
        A JSON schema dictionary
    """
    
    def build_schema_properties(tax_dict: Dict[str, Any], level: int = 1) -> Dict[str, Any]:
        """Recursively build schema properties from taxonomy."""
        properties = {}
        
        for key, value in tax_dict.items():
            if isinstance(value, dict):
                # Nested dictionary - create object schema
                nested_props = build_schema_properties(value, level + 1)
                
                # Build description from examples if available
                descriptions = []
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list) and sub_value:
                        examples = ', '.join(str(v) for v in sub_value[:3])  # Limit examples
                        descriptions.append(f"{sub_key}: e.g., {examples}")
                    elif isinstance(sub_value, dict):
                        # Third level - collect examples
                        for item_key, item_value in sub_value.items():
                            if isinstance(item_value, list) and item_value:
                                examples = ', '.join(str(v) for v in item_value[:3])
                                descriptions.append(f"{sub_key}/{item_key}: e.g., {examples}")
                
                description = f"Description of {key}"
                if descriptions:
                    description += f". Examples: {'; '.join(descriptions[:5])}"  # Limit to 5
                
                properties[key] = {
                    "type": "object",
                    "description": description,
                    "properties": nested_props,
                    "additionalProperties": False
                }
            elif isinstance(value, list) and value:
                # Has examples - use string with description
                examples = ', '.join(str(v) for v in value[:5])  # Limit to 5 examples
                properties[key] = {
                    "type": "string",
                    "description": f"Description of {key}. Examples: {examples}. Write 'None' if not applicable."
                }
            else:
                # No examples - simple string
                properties[key] = {
                    "type": "string",
                    "description": f"Description of {key}. Write 'None' if not applicable."
                }
        
        return properties
    
    # Build the main schema
    properties = build_schema_properties(taxonomy)
    
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False
    }
    
    return schema


def get_structured_taxonomy_prompt_and_schema(taxonomy: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Get both the prompt and JSON schema for structured taxonomy output.
    
    Args:
        taxonomy: The taxonomy dictionary structure
    
    Returns:
        Tuple of (prompt_text, json_schema)
    """
    # Simpler prompt since structure is enforced by schema
    prompt = "Describe this image following the provided taxonomy structure. For each aspect, provide a detailed description. Write 'None' for aspects that don't apply to the image."
    
    schema = create_taxonomy_json_schema(taxonomy)
    
    return prompt, schema


def list_available_prompts(taxonomy: Dict[str, Any] = None):
    """List all available prompts.
    
    Args:
        taxonomy: Optional taxonomy dictionary to include taxonomy-based prompts
    """
    print("Available prompts:")
    print("=================")
    
    print("\nCaption Prompts:")
    for name, prompt_text in CAPTION_PROMPTS.items():
        # Show first line of prompt as description
        first_line = prompt_text.split('\n')[0].strip()
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."
        print(f"  {name}: {first_line}")
    
    if taxonomy:
        print("\nTaxonomy Prompts:")
        print("  TAXONOMY_DEFAULT: List-based taxonomy description")
        print("  TAXONOMY_STRUCTURED: JSON-structured taxonomy description")
        
        # Show preview of taxonomy categories
        print("\n  Taxonomy categories included:")
        for big_cat in taxonomy.keys():
            print(f"    - {big_cat}")


def get_all_caption_prompt_names(taxonomy: Dict[str, Any] = None) -> List[str]:
    """Get a list of all available caption prompt names.
    
    Args:
        taxonomy: Optional taxonomy dictionary to include taxonomy-based prompts
    
    Returns:
        List of prompt names
    """
    prompt_names = list(CAPTION_PROMPTS.keys())
    
    if taxonomy:
        prompt_names.extend(['TAXONOMY_DEFAULT', 'TAXONOMY_STRUCTURED'])
    
    return prompt_names


if __name__ == "__main__":
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="List and preview available caption prompts")
    parser.add_argument(
        "--taxonomy", 
        type=str, 
        default=None,
        help="Path to taxonomy JSON file (e.g., general_taxonomy_v0.json)"
    )
    parser.add_argument(
        "--show-prompt",
        type=str,
        default=None,
        help="Show full text of a specific prompt (e.g., SIMPLE, LONG, TAXONOMY_DEFAULT, TAXONOMY_STRUCTURED)"
    )
    
    args = parser.parse_args()
    
    # Load taxonomy if provided
    taxonomy = None
    if args.taxonomy:
        try:
            with open(args.taxonomy, 'r') as f:
                taxonomy = json.load(f)
            print(f"Loaded taxonomy from: {args.taxonomy}\n")
        except FileNotFoundError:
            print(f"Error: Taxonomy file not found: {args.taxonomy}")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in taxonomy file: {args.taxonomy}")
            exit(1)
    
    # Show specific prompt if requested
    if args.show_prompt:
        prompt_name = args.show_prompt.upper()
        
        if prompt_name in CAPTION_PROMPTS:
            print(f"Prompt: {prompt_name}")
            print("=" * 60)
            print(CAPTION_PROMPTS[prompt_name])
        elif prompt_name == "TAXONOMY_DEFAULT" and taxonomy:
            print(f"Prompt: {prompt_name}")
            print("=" * 60)
            print(create_taxonomy_prompts(taxonomy, prompt_name="default"))
        elif prompt_name == "TAXONOMY_STRUCTURED" and taxonomy:
            print(f"Prompt: {prompt_name}")
            print("=" * 60)
            print(create_taxonomy_prompts(taxonomy, prompt_name="structured"))
        elif prompt_name.startswith("TAXONOMY") and not taxonomy:
            print(f"Error: Taxonomy prompts require --taxonomy argument")
            exit(1)
        else:
            print(f"Error: Unknown prompt name '{args.show_prompt}'")
            print(f"Available prompts: {', '.join(get_all_caption_prompt_names(taxonomy))}")
            exit(1)
    else:
        # List all available prompts
        list_available_prompts(taxonomy)
        
        print("\n" + "=" * 60)
        print(f"Total prompts available: {len(get_all_caption_prompt_names(taxonomy))}")
        print("\nUse --show-prompt <name> to see the full prompt text")
        if not taxonomy:
            print("Use --taxonomy <path> to include taxonomy-based prompts")
