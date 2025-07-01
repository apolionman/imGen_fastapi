import subprocess
import os
import re


def run_flux(prompt: str):
    python_exec = "/app/flux_venv/bin/python"
    script_path = "/app/app/scripts/flux_run.py"

    try:
        result = subprocess.run(
            [python_exec, script_path, "--prompt", prompt],
            cwd="/app/app/scripts",
            check=True,
            capture_output=True,
            text=True
        )

        output = result.stdout
        error_output = result.stderr

        print(f"STDOUT:\n{output}")
        if error_output:
            print(f"STDERR:\n{error_output}")

        match = re.search(r'Image saved to (.+\.png)', output)
        if match:
            image_path = match.group(1).strip()
            return {
                "status": "success",
                "image_path": image_path,
                "stdout": output,
                "stderr": error_output
            }

        if not match:
            return {
                "status": "error",
                "message": "Image path not found in script output",
                "stdout": output,
                "stderr": error_output
            }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": "Subprocess failed",
            "stdout": e.stdout or "",
            "stderr": e.stderr or ""
        }
