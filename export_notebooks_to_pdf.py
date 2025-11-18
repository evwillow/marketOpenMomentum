#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export all Jupyter notebooks to PDF files.

This script finds all .ipynb files in the notebooks/ directory,
executes them, and exports them as PDF files to the notebook_pdfs/ directory.
"""

import os
import subprocess
import sys
from pathlib import Path

# Prevent __pycache__ creation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def export_notebook_to_pdf(notebook_path: Path, output_dir: Path, timeout: int = 300) -> bool:
    """
    Export a single notebook to PDF using jupyter nbconvert.
    
    Args:
        notebook_path: Path to the .ipynb file
        output_dir: Directory where PDF should be saved
        timeout: Maximum time in seconds to wait for conversion
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Converting {notebook_path.name}...", end=" ", flush=True)
        
        # Use webpdf exporter which works better on Windows and is faster
        # Add --no-input to hide code cells if needed, but keep for now
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        
        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "webpdf",
                "--execute",
                "--ExecutePreprocessor.timeout=300",
                "--ExecutePreprocessor.kernel_name=python3",
                "--ExecutePreprocessor.allow_errors=True",
                str(notebook_path),
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        
        if result.returncode == 0:
            # webpdf creates files with .pdf.pdf extension, fix it
            pdf_name = notebook_path.stem + ".pdf"
            double_pdf = output_dir / f"{notebook_path.stem}.pdf.pdf"
            correct_pdf = output_dir / pdf_name
            
            if double_pdf.exists():
                double_pdf.rename(correct_pdf)
            elif (output_dir / pdf_name).exists():
                # Already correct name
                pass
            else:
                # Try to find the PDF file
                pdf_files = list(output_dir.glob(f"{notebook_path.stem}*.pdf"))
                if pdf_files:
                    pdf_files[0].rename(correct_pdf)
            
            print(f"[OK]")
            return True
        else:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            print(f"[FAILED] {error_msg}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Exceeded {timeout}s")
        return False
    except Exception as e:
        print(f"[ERROR] {str(e)[:100]}")
        return False


def main():
    """Main function to export all notebooks."""
    # Get project root (parent of script directory)
    script_dir = Path(__file__).parent
    notebooks_dir = script_dir / "notebooks"
    output_dir = script_dir / "notebook_pdfs"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Find all notebook files
    notebook_files = sorted(notebooks_dir.glob("*.ipynb"))
    
    if not notebook_files:
        print(f"No notebooks found in {notebooks_dir}")
        return 1
    
    print(f"Found {len(notebook_files)} notebook(s) to convert\n")
    
    # Convert each notebook
    success_count = 0
    for notebook in notebook_files:
        if export_notebook_to_pdf(notebook, output_dir):
            success_count += 1
        print()
    
    # Summary
    print("=" * 60)
    print(f"Summary: {success_count}/{len(notebook_files)} notebooks converted successfully")
    print(f"PDFs saved to: {output_dir}")
    
    return 0 if success_count == len(notebook_files) else 1


if __name__ == "__main__":
    sys.exit(main())
