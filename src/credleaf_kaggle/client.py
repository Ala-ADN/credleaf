"""Small Kaggle CLI boundary. Credentials never enter job files or arguments."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


class KaggleCommandError(RuntimeError):
    """A CLI error; rejected means the server explicitly refused the mutation."""
    def __init__(self, message, *, rejected=False):
        super().__init__(message)
        self.rejected = rejected


def credentials(auth_file=None, *, username=None, password=None, auth_type="token"):
    env = dict(os.environ)
    if auth_file:
        value = Path(auth_file).expanduser().read_text().strip()
        if value.startswith("{"):
            value = json.loads(value)
            username, password, auth_type = value["username"], value["key"], "legacy"
        else:
            password = value
    if password:
        for key in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
            env.pop(key, None)
        if auth_type == "legacy":
            if not username:
                raise ValueError("Legacy authentication requires a username")
            env.update(KAGGLE_USERNAME=username, KAGGLE_KEY=password)
            # CLI 2.2.5 otherwise prefers ~/.kaggle/access_token over explicit
            # legacy keys. Its empty-token-file path disables that fallback.
            env["KAGGLE_API_TOKEN"] = os.devnull
        elif auth_type == "token":
            env["KAGGLE_API_TOKEN"] = password
        else:
            raise ValueError("auth_type must be token or legacy")
    return env


class KaggleCLI:
    def __init__(self, env=None, executable=None):
        self.env = dict(os.environ if env is None else env)
        self.executable = (executable or os.environ.get("CREDLEAF_KAGGLE_BIN") or shutil.which("kaggle")
                           or ("/opt/kaggle-cli/bin/kaggle" if Path("/opt/kaggle-cli/bin/kaggle").is_file() else None))
        if not self.executable:
            raise ValueError("Install the isolated CLI: uv tool install --with six kaggle==2.2.5")

    def run(self, *args, timeout=900):
        result = subprocess.run([self.executable, *map(str, args)], env=self.env, capture_output=True,
                                text=True, timeout=timeout)
        output = (result.stdout + "\n" + result.stderr).strip()
        for key in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
            if self.env.get(key):
                output = output.replace(self.env[key], "[redacted]")
        # CLI 2.2.5 prints API-level create/push errors but still exits zero.
        # Inspect those responses before marking a remote resource as created.
        rejection = re.search(r"(?m)^(?:Dataset creation|Kernel push) error:\s*(.+)$", output)
        if rejection:
            raise KaggleCommandError(f"Kaggle {' '.join(map(str, args[:2]))} rejected: {rejection.group(1)}",
                                     rejected=True)
        if result.returncode:
            raise KaggleCommandError(f"Kaggle {' '.join(map(str, args[:2]))} failed: {output[-3000:]}")
        return output

    def status(self, kernel):
        output = self.run("kernels", "status", kernel, timeout=90)
        match = re.search(r'has status "(?:\w+\.)?(\w+)"', output, re.I)
        if not match:
            raise ValueError("Unrecognized Kaggle kernel status: " + output[-500:])
        status = match.group(1).lower()
        status = {"cancel_requested": "running", "cancel_acknowledged": "cancelled", "new_script": "queued"}.get(status, status)
        if status not in {"complete", "running", "queued", "error", "cancelled", "canceled"}:
            raise ValueError("Unrecognized Kaggle kernel status: " + status)
        return status
