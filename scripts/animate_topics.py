"""Render a growing topic tree and named article-count curves from notebook 03."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import pandas as pd

from analysis.topic_animation import prepare_topic_tree, create_topic_tree_animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed/topic_evolution")
    parser.add_argument("--gif", type=Path, default=ROOT / "notebooks/animation_out/semantic_evolution.gif")
    parser.add_argument("--labels", type=Path, default=ROOT / "notebooks/topic_tree_labels.json")
    parser.add_argument("--max-topics", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=.55)
    args = parser.parse_args()
    def read(name):
        return pd.read_json(args.input_dir / f"{name}.jsonl", lines=True)
    names = json.loads(args.labels.read_text()) if args.labels.exists() else {}
    tree = prepare_topic_tree(read("topics/nodes"), read("topics/edges"), read("topics/membership"),
                              read("chunk_provenance"), read("topics/coverage"),
                              threshold=args.threshold, max_tracks=args.max_topics, names=names)
    story = create_topic_tree_animation(tree)
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    story.animation.save(args.gif, writer=PillowWriter(fps=1.5), dpi=90)
    story.render_frame(len(story.windows) - 1)
    story.figure.savefig(args.gif.with_suffix(".png"), dpi=110, facecolor=story.figure.get_facecolor())
    tree.counts.to_csv(args.gif.with_suffix(".csv"), index=False)
    metadata = dict(
        purpose="growing primary-parent topic forest with article volume",
        input_dir=str(args.input_dir.relative_to(ROOT)) if args.input_dir.is_relative_to(ROOT) else str(args.input_dir),
        threshold=args.threshold, displayed_lineages=len(tree.tracks), total_lineages=tree.total_tracks,
        frames=len(tree.windows), article_unit="distinct captured source-page URL without fragment per topic/month",
        article_overlap="one page may belong to multiple topics; repeated chunks/captures counted once",
        lineage_rule="strongest adjacent-window predecessor; best child continues; other children branch",
        selection="article-months times square root of observed months; include every parent lineage",
        labels=tree.tracks[["track_id", "name", "parent_track"]].astype(object).where(
            pd.notna(tree.tracks[["track_id", "name", "parent_track"]]), None).to_dict("records"),
        label_origin=("explicit display overrides; other names from root topic keywords" if names
                      else "automatic names from root topic keywords"),
        limitation="retrospective display projection; weaker edges/merges omitted; first observation is not necessarily birth",
    )
    args.gif.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(story.figure)
    print(f"Saved {args.gif}: {len(story.windows)} months, {len(tree.tracks)} named lineages, {len(tree.nodes)} snapshots")
    print(tree.tracks[["track_id", "name", "parent_track"]].to_string(index=False))


if __name__ == "__main__":
    main()
