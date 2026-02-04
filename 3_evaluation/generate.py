# Script para geracao de texto com modelo treinado
# Projeto Seriguela - Geracao interativa de expressoes simbolicas

import argparse
import os
import sys
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes.expression import Expression


class ExpressionStoppingCriteria(StoppingCriteria):
    """Stop generation at natural expression boundaries."""
    def __init__(self, tokenizer, stop_sequences):
        self.tokenizer = tokenizer
        self.stop_ids = [tokenizer.encode(seq, add_special_tokens=False)
                        for seq in stop_sequences]

    def __call__(self, input_ids, scores, **kwargs):
        # Check if any stop sequence appears in generated text
        for stop_ids in self.stop_ids:
            if len(stop_ids) > 0 and len(input_ids[0]) >= len(stop_ids):
                if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                    return True
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate expressions with a trained model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (local or HuggingFace Hub)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model for PEFT (if model_path is adapter)")

    # Prompt building arguments
    parser.add_argument("--num_vars", type=int, default=3,
                        help="Number of variables (e.g., 3 for x_1, x_2, x_3)")
    parser.add_argument("--operators", type=str, default="+,-,*,/,sin,cos",
                        help="Comma-separated operators (e.g., '+,-,*,/,sin,cos,log,sqrt,exp')")
    parser.add_argument("--constants", type=str, default="C",
                        help="Constant symbol (default: C)")
    parser.add_argument("--format", type=str, default="infix", choices=["infix", "prefix"],
                        help="Expression format (infix or prefix)")

    # Custom prompt
    parser.add_argument("--custom_prompt", type=str, default=None,
                        help="Use a custom prompt instead of building one")

    # Generation parameters
    parser.add_argument("--num_generations", type=int, default=5,
                        help="Number of expressions to generate")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher = more diverse)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")

    # Behavior
    parser.add_argument("--validate", action="store_true",
                        help="Validate generated expressions")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def build_prompt(num_vars: int, operators: list, constants: str = "C",
                 format_type: str = "infix") -> str:
    """Build a prompt for expression generation."""
    # Build variables string
    vars_list = [f"x_{i}" for i in range(1, num_vars + 1)]
    vars_str = ", ".join(vars_list)

    # Build operators string
    ops_str = ", ".join(operators)

    # Build prompt based on format
    if format_type == "infix":
        prompt = f"""Variables: {vars_str}
Operators: {ops_str}
Constants: {constants}
Expression: <|startofex|>"""
    else:  # prefix
        prompt = f"""Variables: {vars_str}
Operators: {ops_str}
Constants: {constants}
Prefix Expression: <|startofex|>"""

    return prompt


def load_model_and_tokenizer(model_path: str, base_model: str = None, device: str = "auto"):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if this is a PEFT model
    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json")) if os.path.isdir(model_path) else False

    if is_peft or base_model:
        # Load base model first
        base = base_model or "gpt2"
        print(f"Loading base model: {base}")
        model = AutoModelForCausalLM.from_pretrained(base)
        model.resize_token_embeddings(len(tokenizer))

        # Load PEFT adapter
        print("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_expressions(model, tokenizer, prompt: str, device: str,
                         num_generations: int = 5, max_new_tokens: int = 64,
                         temperature: float = 0.7, top_p: float = 0.9,
                         top_k: int = 50):
    """Generate expressions from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get special token IDs - prefer <|endofex|> as EOS
    end_token_id = tokenizer.convert_tokens_to_ids("<|endofex|>")
    if end_token_id == tokenizer.unk_token_id:
        print("Warning: <|endofex|> not in tokenizer, using default eos_token_id")
        end_token_id = tokenizer.eos_token_id

    # Create stopping criteria to stop at natural expression boundaries (backup)
    stop_sequences = ["\nvars:", "\nVariables:", "\nOperators:", "\n\n"]
    stopping_criteria = StoppingCriteriaList([
        ExpressionStoppingCriteria(tokenizer, stop_sequences)
    ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=end_token_id,  # Use <|endofex|> as EOS
            stopping_criteria=stopping_criteria,  # Keep as backup
        )

    generated = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return generated


def extract_expression(output: str) -> str:
    """Extract the expression from generated output."""
    # Try marker-based first
    start_marker = "<|startofex|>"
    end_marker = "<|endofex|>"

    if start_marker in output and end_marker in output:
        start_idx = output.find(start_marker) + len(start_marker)
        end_idx = output.find(end_marker)
        if start_idx < end_idx:
            return output[start_idx:end_idx].strip()

    # Fallback: Extract first complete expression after start marker
    if start_marker in output:
        start_idx = output.find(start_marker) + len(start_marker)
        remaining = output[start_idx:].strip()

        # Split at common boundaries
        for boundary in ["\nvars:", "\nVariables:", "\nOperators:", "\n\n", "<|endoftext|>"]:
            if boundary in remaining:
                remaining = remaining.split(boundary)[0].strip()
                break

        # Remove any trailing incomplete text - take just the first line
        remaining = remaining.split("\n")[0].strip()

        # Limit length if unreasonably long
        if len(remaining) > 150:
            remaining = remaining[:150]

        return remaining

    # Last resort: look for "expr:" or "Expression:" pattern
    match = re.search(r'(?:expr|Expression):\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Give up: return first line, limited length
    first_line = output.strip().split("\n")[0]
    return first_line[:100] if len(first_line) > 100 else first_line


def validate_expression(expr_str: str, is_prefix: bool = False) -> dict:
    """Validate an expression."""
    result = {
        "valid": False,
        "error": None,
        "sympy_str": None
    }

    if not expr_str:
        result["error"] = "Empty expression"
        return result

    try:
        expr = Expression(expr_str, is_prefix=is_prefix)
        result["valid"] = True
        result["sympy_str"] = expr.sympy_str()
    except Exception as e:
        result["error"] = str(e)

    return result


def print_generation_result(idx: int, expr_str: str, validation: dict = None):
    """Print a formatted generation result."""
    print(f"\n[{idx + 1}] {expr_str}")
    if validation:
        if validation["valid"]:
            print(f"    Status: VALID")
            if validation["sympy_str"] != expr_str:
                print(f"    Sympy: {validation['sympy_str']}")
        else:
            print(f"    Status: INVALID - {validation['error']}")


def interactive_mode(model, tokenizer, device, args):
    """Run in interactive mode."""
    print("\n" + "="*60)
    print("SERIGUELA - Interactive Expression Generator")
    print("="*60)
    print("Commands:")
    print("  /vars N      - Set number of variables (e.g., /vars 3)")
    print("  /ops +,-,*   - Set operators (e.g., /ops +,-,*,sin)")
    print("  /format X    - Set format (infix or prefix)")
    print("  /temp T      - Set temperature (e.g., /temp 0.8)")
    print("  /n N         - Set number of generations (e.g., /n 10)")
    print("  /prompt      - Show current prompt")
    print("  /gen         - Generate with current settings")
    print("  /custom TEXT - Use custom prompt")
    print("  /quit        - Exit")
    print("="*60)

    # Current settings
    settings = {
        "num_vars": args.num_vars,
        "operators": args.operators.split(","),
        "format": args.format,
        "temperature": args.temperature,
        "num_generations": args.num_generations,
        "custom_prompt": None
    }

    is_prefix = settings["format"] == "prefix"

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd == "/quit" or cmd == "/exit":
                print("Goodbye!")
                break

            elif cmd == "/vars" and arg:
                try:
                    settings["num_vars"] = int(arg)
                    print(f"Variables set to {settings['num_vars']}")
                except ValueError:
                    print("Invalid number")

            elif cmd == "/ops" and arg:
                settings["operators"] = [op.strip() for op in arg.split(",")]
                print(f"Operators set to: {settings['operators']}")

            elif cmd == "/format" and arg:
                if arg.lower() in ["infix", "prefix"]:
                    settings["format"] = arg.lower()
                    is_prefix = settings["format"] == "prefix"
                    print(f"Format set to {settings['format']}")
                else:
                    print("Invalid format. Use 'infix' or 'prefix'")

            elif cmd == "/temp" and arg:
                try:
                    settings["temperature"] = float(arg)
                    print(f"Temperature set to {settings['temperature']}")
                except ValueError:
                    print("Invalid temperature")

            elif cmd == "/n" and arg:
                try:
                    settings["num_generations"] = int(arg)
                    print(f"Number of generations set to {settings['num_generations']}")
                except ValueError:
                    print("Invalid number")

            elif cmd == "/prompt":
                prompt = build_prompt(
                    settings["num_vars"],
                    settings["operators"],
                    "C",
                    settings["format"]
                )
                print(f"\nCurrent prompt:\n{prompt}")

            elif cmd == "/custom" and arg:
                settings["custom_prompt"] = arg
                print(f"Custom prompt set")

            elif cmd == "/gen":
                # Generate
                if settings["custom_prompt"]:
                    prompt = settings["custom_prompt"]
                else:
                    prompt = build_prompt(
                        settings["num_vars"],
                        settings["operators"],
                        "C",
                        settings["format"]
                    )

                print(f"\nGenerating {settings['num_generations']} expressions...")
                print("-"*40)

                outputs = generate_expressions(
                    model, tokenizer, prompt, device,
                    num_generations=settings["num_generations"],
                    temperature=settings["temperature"],
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens
                )

                valid_count = 0
                for i, output in enumerate(outputs):
                    expr_str = extract_expression(output)
                    validation = validate_expression(expr_str, is_prefix)
                    print_generation_result(i, expr_str, validation)
                    if validation["valid"]:
                        valid_count += 1

                print("-"*40)
                print(f"Valid: {valid_count}/{len(outputs)}")

            else:
                print(f"Unknown command: {cmd}")

        else:
            # Treat as custom prompt and generate
            prompt = user_input if "<|startofex|>" in user_input else user_input + " <|startofex|>"

            print(f"\nGenerating {settings['num_generations']} expressions...")
            print("-"*40)

            outputs = generate_expressions(
                model, tokenizer, prompt, device,
                num_generations=settings["num_generations"],
                temperature=settings["temperature"],
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens
            )

            valid_count = 0
            for i, output in enumerate(outputs):
                expr_str = extract_expression(output)
                validation = validate_expression(expr_str, is_prefix) if args.validate else None
                print_generation_result(i, expr_str, validation)
                if validation and validation["valid"]:
                    valid_count += 1

            if args.validate:
                print("-"*40)
                print(f"Valid: {valid_count}/{len(outputs)}")


def main():
    args = parse_args()

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path, args.base_model, args.device
    )

    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, device, args)
        return

    # Build or use custom prompt
    if args.custom_prompt:
        prompt = args.custom_prompt
    else:
        operators = [op.strip() for op in args.operators.split(",")]
        prompt = build_prompt(
            args.num_vars,
            operators,
            args.constants,
            args.format
        )

    print("\n" + "="*60)
    print("SERIGUELA - Expression Generator")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Format: {args.format}")
    print(f"Temperature: {args.temperature}")
    print("-"*60)
    print("Prompt:")
    print(prompt)
    print("-"*60)

    # Generate
    is_prefix = args.format == "prefix"

    outputs = generate_expressions(
        model, tokenizer, prompt, device,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )

    print(f"\nGenerated {len(outputs)} expressions:")
    print("-"*60)

    valid_count = 0
    for i, output in enumerate(outputs):
        expr_str = extract_expression(output)
        validation = validate_expression(expr_str, is_prefix) if args.validate else None
        print_generation_result(i, expr_str, validation)
        if validation and validation["valid"]:
            valid_count += 1

    if args.validate:
        print("-"*60)
        print(f"\nSummary: {valid_count}/{len(outputs)} valid expressions ({valid_count/len(outputs)*100:.1f}%)")

    print("="*60)


if __name__ == "__main__":
    main()
