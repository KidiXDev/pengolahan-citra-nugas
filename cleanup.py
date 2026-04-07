"""
Jobsheet Code Cleaner
=====================
Downloads Qwen2.5-Coder-1.5B-Instruct GGUF from HuggingFace and uses it
to strip line numbers and fix Python indentation from jobsheet code.

Requirements:
    pip install llama-cpp-python huggingface-hub rich

For GPU acceleration (optional):
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

Usage:`
    python jobsheet_cleaner.py                  # interactive mode
    python jobsheet_cleaner.py -f input.txt     # read from file
    python jobsheet_cleaner.py -o output.py     # save to file
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Depdency Check
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERROR] huggingface-hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)

try:
    from llama_cpp import Llama
except ImportError:
    print("[ERROR] llama-cpp-python not installed. Run: pip install llama-cpp-python")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich import print as rprint

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# Config

HF_REPO = "unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF"
HF_FILENAME = "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"
MODEL_DIR = Path.home() / ".cache" / "jobsheet_cleaner"

SYSTEM_PROMPT = """You are a Python code formatter assistant.
The user will give you Python code extracted from a PDF or image jobsheet.
The code may have:
- Line number prefixes like "03:", "14 :", "07 :" at the start of lines
- Broken or inconsistent indentation (wrong tabs/spaces, bad nesting)
- Multiple statements merged onto one line with a number in between

Your job:
1. Remove ALL line number prefixes (digits followed by colon).
2. Reconstruct correct Python indentation using 4 spaces per indent level.
3. Infer the correct nesting from the code's logical structure (if/for/while/def/class/with/try).
4. DO not changes the code logic, only fix the indentation.
5. Make sure all variable declaration is correct.
6. Return ONLY the fixed Python code. No explanation, no markdown fences, no commentary."""


# Helpers

console = Console() if HAS_RICH else None


def log(msg: str, style: str = ""):
    if HAS_RICH:
        console.print(msg, style=style)
    else:
        print(msg)


def pre_clean(text: str) -> str:
    """
    Best-effort regex pre-pass before sending to AI.
    Splits mid-line merged lines (e.g. "...) 14: # comment")
    and strips line number prefixes.
    """
    # Split on line-number markers that appear mid-line
    text = re.sub(r"(?<!\n)\s+(\d{1,3})\s*:\s", r"\n\1: ", text)

    lines = []
    for line in text.splitlines():
        # Strip leading line number: "  03 : " or "14:"
        cleaned = re.sub(r"^\s*\d{1,3}\s*:\s?", "", line)
        lines.append(cleaned)
    return "\n".join(lines)


def download_model(hf_token: str | None) -> Path:
    """Download GGUF model from HuggingFace if not already cached."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_DIR / HF_FILENAME

    if local_path.exists():
        log(
            f"[bold green]✓[/] Model already cached at {local_path}"
            if HAS_RICH
            else f"Model already cached at {local_path}"
        )
        return local_path

    log(
        f"[bold yellow]↓[/] Downloading {HF_FILENAME} from {HF_REPO}..."
        if HAS_RICH
        else f"Downloading {HF_FILENAME}..."
    )
    log("  This is ~1 GB and only happens once.", "dim" if HAS_RICH else "")

    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        local_dir=str(MODEL_DIR),
        token=hf_token,
    )
    log(
        f"[bold green]✓[/] Saved to {downloaded}"
        if HAS_RICH
        else f"Saved to {downloaded}"
    )
    return Path(downloaded)


def load_model(model_path: Path) -> Llama:
    """Load GGUF model with llama-cpp-python."""
    log(
        "[bold yellow]⚙[/] Loading model into memory..."
        if HAS_RICH
        else "Loading model..."
    )
    llm = Llama(
        model_path=str(model_path),
        n_ctx=4096,  # context window
        n_threads=os.cpu_count(),
        n_gpu_layers=-1,  # use GPU if available, else 0
        verbose=False,
    )
    log("[bold green]✓[/] Model loaded.\n" if HAS_RICH else "Model loaded.\n")
    return llm


def fix_code(llm: Llama, raw_code: str) -> str:
    """Send pre-cleaned code to the model and return fixed Python."""
    pre_cleaned = pre_clean(raw_code)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Fix this jobsheet code:\n\n{pre_cleaned}"},
    ]

    log(
        "[bold blue]⟳[/] AI is fixing indentation..."
        if HAS_RICH
        else "AI is fixing indentation..."
    )

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=2048,
        temperature=0.1,  # low temp = more deterministic output
        repeat_penalty=1.1,
    )

    result = response["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if model adds them anyway
    result = re.sub(r"^```python\s*\n?", "", result)
    result = re.sub(r"\n?```$", "", result)
    return result.strip()


def get_multiline_input() -> str:
    """Read multiline paste from stdin until user types END on its own line."""
    if HAS_RICH:
        console.print(
            "\n[bold]Paste your jobsheet code below.[/]\n"
            "Type [bold cyan]END[/] on its own line when done, then press Enter.\n",
        )
    else:
        print("\nPaste your jobsheet code below.")
        print("Type END on its own line when done, then press Enter.\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def display_result(code: str):
    if HAS_RICH:
        console.print(
            Panel(
                Syntax(code, "python", theme="monokai", line_numbers=True),
                title="[bold green]Fixed Python Code[/]",
                border_style="green",
            )
        )
    else:
        print("\n=== Fixed Python Code ===")
        print(code)
        print("=========================\n")


# Main


def main():
    parser = argparse.ArgumentParser(
        description="Jobsheet Code Cleaner — strips line numbers and fixes Python indentation using local AI."
    )
    parser.add_argument(
        "--token",
        "-t",
        help="HuggingFace API token (or set HF_TOKEN env var). Required for first download.",
        default=os.environ.get("HF_TOKEN"),
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Read input code from a text file instead of interactive paste.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save fixed code to this file (e.g. output.py).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Only strip line numbers (regex), skip AI indentation fix.",
    )
    args = parser.parse_args()

    if HAS_RICH:
        console.rule("[bold]Jobsheet Code Cleaner[/]")
    else:
        print("=" * 40)
        print("  Jobsheet Code Cleaner")
        print("=" * 40)

    # Download + Load Model
    if not args.no_ai:
        if not args.token:
            log(
                "[yellow]⚠[/]  No HuggingFace token found.\n"
                "   Set HF_TOKEN env var or pass --token.\n"
                "   If the model is already cached this is fine."
                if HAS_RICH
                else "Warning: No HuggingFace token. Set HF_TOKEN env var or pass --token."
            )
        model_path = download_model(args.token)
        llm = load_model(model_path)

    # Get Input
    if args.file:
        raw = Path(args.file).read_text(encoding="utf-8")
        log(
            f"[dim]Read {len(raw.splitlines())} lines from {args.file}[/]"
            if HAS_RICH
            else f"Read from {args.file}"
        )
    else:
        raw = get_multiline_input()

    if not raw.strip():
        log("[red]No input provided. Exiting.[/]" if HAS_RICH else "No input. Exiting.")
        sys.exit(0)

    # Process
    if args.no_ai:
        fixed = pre_clean(raw)
        log(
            "[dim]Regex-only mode: line numbers stripped.[/]"
            if HAS_RICH
            else "Regex-only: line numbers stripped."
        )
    else:
        fixed = fix_code(llm, raw)

    # Output
    display_result(fixed)

    if args.output:
        Path(args.output).write_text(fixed + "\n", encoding="utf-8")
        log(
            f"\n[bold green]✓[/] Saved to [cyan]{args.output}[/]"
            if HAS_RICH
            else f"\nSaved to {args.output}"
        )

    # Loop for another paste
    if not args.file and not args.no_ai:
        while True:
            if HAS_RICH:
                again = Prompt.ask(
                    "\n[bold]Fix another snippet?[/]", choices=["y", "n"], default="n"
                )
            else:
                again = (
                    input("\nFix another snippet? (y/n) [n]: ").strip().lower() or "n"
                )

            if again != "y":
                break

            raw = get_multiline_input()
            if not raw.strip():
                break
            fixed = fix_code(llm, raw)
            display_result(fixed)

            if args.output:
                Path(args.output).write_text(fixed + "\n", encoding="utf-8")
                log(
                    f"[bold green]✓[/] Saved to [cyan]{args.output}[/]"
                    if HAS_RICH
                    else f"Saved to {args.output}"
                )


if __name__ == "__main__":
    main()
