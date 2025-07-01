import subprocess
import os
import re

def run_flux(prompt: str):
    python_exec = "/app/flux_venv/bin/python"
    script_path = "/app/app/scripts/flux_run.py"

    try:
        result = subprocess.run(
            [python_exec, script_path, "--prompt", prompt],
            cwd="/app/flux_runner",
            check=True,
            capture_output=True,
            text=True
        )

        output = result.stdout
        print(output)

        # Parse image path
        match = re.search(r'OUTPUT::(.+\.png)', output)
        if match:
            image_path = match.group(1).strip()
            return {"status": "success", "image_path": image_path}

        return {"status": "error", "message": "Image path not found"}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": e.stderr}
