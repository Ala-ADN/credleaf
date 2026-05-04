"""Print a summary of each configured Archive-It collection.
"""
from __future__ import annotations

import json

from config import COVID_COLLECTIONS
from ingest import ArchiveItClient


def main() -> None:
    with ArchiveItClient() as client:
        for cid, cfg in COVID_COLLECTIONS.items():
            print("=" * 72)
            print(f"Collection {cid}: {cfg.name}  [credibility={cfg.credibility_tier}]")
            print("=" * 72)

            coll = client.get_collection(cid)
            warc_gb = coll.total_warc_bytes / 1e9
            print(
                f"  active seeds : {coll.num_active_seeds}\n"
                f"  inactive     : {coll.num_inactive_seeds}\n"
                f"  total WARC   : {warc_gb:,.1f} GB\n"
                f"  last crawl   : {coll.last_crawl_date}"
                f"  created      : {coll.created_date}\n"
                f"  metadata     : {json.dumps(coll.metadata, sort_keys=True, indent=2) or '-'}"
            )

            seeds = list(client.iter_seeds(cid, page_size=50, max_seeds=5))
            print(f"\n  first {len(seeds)} seeds:")
            for s in seeds:
                print(f"    [{s.id}] {s.url}")

            if seeds:
                target = seeds[0].canonical_url or seeds[0].url
                caps = list(
                    client.iter_captures(cid, target, match_type="exact", limit=3)
                )
                print(f"\n  CDX sample for {target!r}: {len(caps)} captures")
                for c in caps:
                    print(
                        f"    {c.timestamp}  {c.status or '-':>3}  "
                        f"{c.mimetype or '-':<20}  {c.length or 0:>8}B"
                    )
            print()


if __name__ == "__main__":
    main()
