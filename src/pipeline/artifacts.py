"""Immutable, checksummed stage artifacts with atomic directory publication."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from pipeline.inventory import describe
from pipeline.spec import required_files


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def implementation(root):
    files = [root / "pyproject.toml", root / "uv.lock"]
    files += [path for folder in ("src", "scripts", "dags") for path in (root / folder).rglob("*.py")]
    return digest({str(path.relative_to(root)): file_hash(path) for path in sorted(files) if path.is_file()})


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.base = self.root / "data/pipeline/artifacts"

    def directory(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z_]+-[0-9a-f]{32}", identifier):
            raise ValueError("Invalid artifact ID")
        return self.base / identifier

    def load(self, reference):
        identifier = reference if isinstance(reference, str) else reference["id"]
        directory = self.directory(identifier)
        if directory.is_symlink():
            raise ValueError("Artifact directories cannot be symlinks")
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["id"] != identifier or manifest["format"] != 1:
            raise ValueError("Invalid artifact manifest")
        signature = {key: manifest[key] for key in ("stage", "collection_id", "config", "inputs", "implementation", "nonce")}
        if identifier != f"{manifest['stage']}-{digest(signature)[:32]}":
            raise ValueError("Artifact signature does not match its ID")
        if isinstance(reference, dict) and reference["sha256"] != file_hash(path):
            raise ValueError(f"Manifest changed: {identifier}")
        if isinstance(reference, dict) and any(reference[key] != manifest[key] for key in ("stage", "collection_id")):
            raise ValueError("Artifact reference scope differs from its manifest")
        names = [item["path"] for item in manifest["files"]]
        if len(set(names)) != len(names) or not set(required_files(manifest["stage"], manifest["collection_id"], manifest["config"])) <= set(names):
            raise ValueError("Incomplete artifact inventory")
        for item in manifest["files"]:
            file = directory / "files" / item["path"]
            if file.is_symlink() or not file.resolve().is_relative_to((directory / "files").resolve()):
                raise ValueError("Artifact path escapes its directory")
            if not file.is_file() or file_hash(file) != item["sha256"]:
                raise ValueError(f"Artifact missing or changed: {identifier}/{item['path']}")
        return manifest

    def reference(self, manifest):
        return dict(id=manifest["id"], stage=manifest["stage"], collection_id=manifest["collection_id"],
                    sha256=file_hash(self.directory(manifest["id"]) / "manifest.json"))

    def payload(self, reference):
        self.load(reference)
        return self.directory(reference if isinstance(reference, str) else reference["id"]) / "files"

    def verify_tree(self, reference, seen=None):
        seen = set() if seen is None else seen
        token = digest(reference)
        if token in seen:
            return
        manifest = self.load(reference)
        seen.add(token)
        for parent in manifest["inputs"]:
            self.verify_tree(parent, seen)

    def bundle(self, reference, kind=None):
        self.verify_tree(reference)
        manifest = self.load(reference)
        if manifest["stage"] != "publish":
            raise ValueError("Expected a published snapshot ID")
        bundle = json.loads((self.payload(reference) / "snapshot.json").read_text())
        if bundle["kind"] != manifest["config"]["kind"] or [ref for ref in bundle["artifacts"] if ref.get("id")] != manifest["inputs"]:
            raise ValueError("Published snapshot differs from its pinned inputs")
        if kind is not None and bundle["kind"] != kind:
            raise ValueError(f"Expected a {kind} snapshot")
        return bundle

    @contextmanager
    def _lock(self, identifier):
        locks = self.base.parent / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / identifier).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def build(self, stage, cid, config, inputs, code, producer, *, nonce=None):
        inputs = [ref for ref in inputs if ref.get("id")]
        seen = set()
        for ref in inputs:
            self.verify_tree(ref, seen)
        signature = dict(stage=stage, collection_id=cid, config=config, inputs=inputs, implementation=code, nonce=nonce)
        identifier = f"{stage}-{digest(signature)[:32]}"
        destination = self.directory(identifier)
        self.base.mkdir(parents=True, exist_ok=True)
        with self._lock(identifier):
            if destination.exists():
                return self.reference(self.load(identifier)), True
            with tempfile.TemporaryDirectory(prefix=f".{identifier}-", dir=self.base) as temporary:
                staging = Path(temporary)
                files = staging / "files"
                files.mkdir()
                stats = producer(files)
                seen = set()
                for ref in inputs:
                    self.verify_tree(ref, seen)
                missing = [name for name in required_files(stage, cid, config) if not (files / name).is_file()]
                if missing:
                    raise ValueError(f"Missing {stage} outputs: {missing}")
                inventory = []
                for path in sorted(files.rglob("*")):
                    if path.is_symlink():
                        raise ValueError("Artifact outputs cannot be symlinks")
                    if path.is_file():
                        item = describe(path, files)
                        item["sha256"] = file_hash(path)
                        with path.open("rb") as source:
                            os.fsync(source.fileno())
                        inventory.append(item)
                for directory in sorted((p for p in files.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                    sync_directory(directory)
                sync_directory(files)
                manifest = dict(format=1, id=identifier, **signature, files=inventory, stats=stats,
                                created_at=datetime.now(timezone.utc).isoformat())
                write_json(staging / "manifest.json", manifest)
                staging.rename(destination)
                sync_directory(self.base)
            return self.reference(manifest), False

    def reuse(self, reference, stage, cid, config, inputs, code):
        self.verify_tree(reference)
        manifest = self.load(reference)
        expected = dict(stage=stage, collection_id=cid, config=config,
                        inputs=[ref for ref in inputs if ref.get("id")], implementation=code)
        for key, value in expected.items():
            if manifest[key] != value:
                raise ValueError(f"Incompatible reused {stage}: {key} differs; build it instead")
        return self.reference(manifest)


def lookup(bundle, stage, cid=None):
    for ref in bundle["artifacts"]:
        if ref["stage"] == stage and ref["collection_id"] == cid:
            if not ref.get("id"):
                break
            return ref
    raise ValueError(f"Snapshot has no {stage} artifact for collection {cid}")
