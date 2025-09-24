import os
import argparse
from pathlib import Path

exclude = [
    "**/*.zip",
    "**/create_context.py",
    "**/__pycache__/**",
    "**/node_modules/**",
    "**/.git/**",
    "**/.env",
    "**/dist/**",
    "**/public/**",
    "**/*.db",
    "**/caption_translations",
    "**/create.py",
    "**/languages.json",
    "**/requirements.txt",
    "**/*.md",
    "**/.gitignore",
    "**/package*.json",
    "**/vite.config.ts",
    "**/*key**",
    "**/*.svg",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.ipynb",
    "**/*.log",
]

formatting = {
    ".py": "```python\n# {file}\n{content}\n```",
    ".js": "```javascript\n// {file}\n{content}\n```",
    ".ts": "```typescript\n// {file}\n{content}\n```",
    ".tsx": "```typescript\n// {file}\n{content}\n```",
    ".json": "```json\n// {file}\n{content}\n```",
    ".html": "```html\n<!-- {file} -->\n{content}\n```",
    ".css": "```css\n/* {file} */\n{content}\n```",
}

def main(input_dir: Path, output_file: Path):
    files = []
    # Walk through the directory and collect files, excluding specified patterns.
    for root, dirs, filenames in os.walk(input_dir):
        # Exclude directories based on the exclude patterns.
        dirs[:] = [d for d in dirs if not any(Path(root, d).match(pattern) for pattern in exclude)]
        
        for filename in filenames:
            file_path = Path(root) / filename
            # Exclude files based on the exclude patterns.
            if not any(file_path.match(pattern) for pattern in exclude):
                files.append(file_path)

    print(f"Found {len(files)} files to process.")
    # Write the context to the output file.
    with open(output_file, "w", encoding="utf-8") as f:
        for file_path in files:
            # Read the content of the file.
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read().strip()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
            
            # Determine the file extension and format the content accordingly.
            ext = file_path.suffix.lower()
            if ext in formatting:
                relative_path = file_path.relative_to(input_dir)
                formatted_content = formatting[ext].format(file=relative_path, content=content)
                f.write(formatted_content + "\n\n")
            else:
                print(f"Unsupported file type: {file_path}")
                continue


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Create a context file from a directory.")
    # Optional input_dir argument.
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path(os.getcwd()),
        help="The directory to scan for files. Defaults to the current working directory."
    )
    # Optional output_file argument.
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        default=Path("context.md"),
        help="The output file to write the context to. Defaults to 'context.txt'."
    )
    args = parser.parse_args()

    # Ensure input_dir is a directory.
    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory '{args.input_dir}' does not exist or is not a directory.")
    
    main(args.input_dir, args.output_file)
