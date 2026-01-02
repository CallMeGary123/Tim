import argparse
import shutil
from pathlib import Path
import sys

def remove_corrupt_raw_and_processed(data_dir: Path, corrupt_name: str, dry_run: bool):
    # Remove raw/images/<corrupt_name>
    raw_path = data_dir / 'raw' / 'images' / corrupt_name
    if raw_path.exists():
        if dry_run:
            print(f"[DRY RUN] Would remove directory: {raw_path}")
        else:
            print(f"Removing directory: {raw_path}")
            shutil.rmtree(raw_path)
    else:
        print(f"No raw folder found at: {raw_path}")

    # Remove any files or dirs in processed that contain the corrupt_name
    processed_root = data_dir / 'processed'
    if not processed_root.exists():
        print(f"No processed folder at: {processed_root}")
        return

    for p in processed_root.rglob('*'):
        if corrupt_name in p.name:
            if p.is_dir():
                if dry_run:
                    print(f"[DRY RUN] Would remove directory: {p}")
                else:
                    print(f"Removing directory: {p}")
                    shutil.rmtree(p)
            elif p.is_file():
                if dry_run:
                    print(f"[DRY RUN] Would remove file: {p}")
                else:
                    print(f"Removing file: {p}")
                    p.unlink()

def rename_instances(data_dir: Path, old_fragment: str, new_fragment: str, dry_run: bool):
    # Collect paths containing old_fragment, sort by depth (deepest first)
    candidates = [p for p in data_dir.rglob('*') if old_fragment in p.name]
    candidates.sort(key=lambda p: len(p.parts), reverse=True)

    for p in candidates:
        new_name = p.name.replace(old_fragment, new_fragment)
        target = p.with_name(new_name)

        if target.exists():
            # If both directories, merge contents
            if p.is_dir() and target.is_dir():
                for child in p.iterdir():
                    dest = target / child.name
                    if dest.exists():
                        # avoid overwriting: add _dup suffix
                        dest = target / (child.stem + '_dup' + child.suffix)
                    if dry_run:
                        print(f"[DRY RUN] Would move {child} -> {dest}")
                    else:
                        print(f"Moving {child} -> {dest}")
                        shutil.move(str(child), str(dest))
                if dry_run:
                    print(f"[DRY RUN] Would remove empty directory {p}")
                else:
                    try:
                        p.rmdir()
                    except OSError:
                        pass
            else:
                # name conflict: rename source to a unique name
                alt = target.with_name(target.stem + '_dup' + target.suffix)
                if dry_run:
                    print(f"[DRY RUN] Would rename (conflict) {p} -> {alt}")
                else:
                    print(f"Renaming (conflict) {p} -> {alt}")
                    p.rename(alt)
        else:
            if dry_run:
                print(f"[DRY RUN] Would rename {p} -> {target}")
            else:
                print(f"Renaming {p} -> {target}")
                p.rename(target)

def main():
    # default data dir: two levels up from this file + /data (relative to repo root)
    default_data_dir = Path(__file__).resolve().parents[2] / 'data'

    parser = argparse.ArgumentParser(description="Fix artist name issues in dataset")
    parser.add_argument("--data-dir", default=str(default_data_dir),
                        help=f"Root data directory (default: {default_data_dir} relative to this script)")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Omit to do a dry run.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    dry_run = not args.apply

    corrupt_name = "Albrecht_DuΓòá├¬rer"
    old_name = "Albrecht_Du╠êrer"
    new_name = "Albrecht_Dürer"

    print(f"{'DRY RUN - no changes will be made' if dry_run else 'Applying changes'}")
    remove_corrupt_raw_and_processed(data_dir, corrupt_name, dry_run)
    rename_instances(data_dir, old_name, new_name, dry_run)
    print("Done.")

if __name__ == "__main__":
    main()
