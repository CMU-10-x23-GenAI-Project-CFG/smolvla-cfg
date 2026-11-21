"""Summarise placement + first-grasp rates from an eval results folder.

Reads the per-episode trajectory JSONs that the LIBERO-CF evaluator writes
to ``rollouts/<exp>/trajectories/*.json`` and produces the numbers for the
midway tables: instructed / distractor placement rates and a first-grasp
breakdown.

What the saved JSON gives us (per episode):
  * ``per_condition[]`` with ``condition``, ``touched`` (gripper ever
    contacted that object), ``success`` (object placed in goal).
  * ``steps[]`` with ``gripper_qpos`` per timestep (used to find the first
    gripper-close event).

Honest limitation: the evaluator does NOT save which body was contacted at
each step -- contact is only aggregated into the per-condition ``touched``
flag. So we can classify a first grasp as instructed / distractor / neither,
and separate "closed on nothing" (closed but touched no condition object)
from "closed on something", but we cannot name a *third* object the gripper
closed on. The "other" column is therefore reported as "neither (touched
some non-condition object or mis-timed)"; treat "nothing" as the headline
failure the schedules target.

Convention: on ``libero_cf_object`` the first listed condition is the
instructed object and the second is the distractor (matches the benchmark's
condition ordering and negatives.json). Pass --instructed-index / 
--distractor-index to override for other suites.

Usage:
    python scripts/first_grasp.py results/early_k4
    python scripts/first_grasp.py results/early_k4 --json   # machine-readable
"""

import argparse
import json
import pathlib
import statistics


def gripper_closed(gripper_qpos, threshold):
    """Heuristic: the Panda gripper is 'closed' when the finger opening is
    below ``threshold``. gripper_qpos holds the two finger joints; their sum
    is the opening. The default threshold is conservative; tune with
    --close-threshold if a run's gripper scale differs."""
    if not gripper_qpos:
        return False
    return sum(abs(q) for q in gripper_qpos) < threshold


def first_close_step(steps, threshold):
    """Index of the first timestep where the gripper transitions to closed,
    or None if it never closes."""
    was_open = True
    for i, s in enumerate(steps):
        closed = gripper_closed(s.get("gripper_qpos", []), threshold)
        if closed and was_open:
            return i
        was_open = not closed
    return None


def classify_episode(ep, instructed_idx, distractor_idx, threshold):
    """Return a dict of booleans for one episode."""
    per_cond = ep.get("per_condition", [])
    out = {
        "instructed_success": False,
        "distractor_success": False,
        "instructed_touched": False,
        "distractor_touched": False,
        "closed_on_something": False,
        "closed_on_nothing": False,
        "never_closed": False,
    }
    if len(per_cond) > max(instructed_idx, distractor_idx):
        instr, distr = per_cond[instructed_idx], per_cond[distractor_idx]
        out["instructed_success"] = bool(instr.get("success"))
        out["distractor_success"] = bool(distr.get("success"))
        out["instructed_touched"] = bool(instr.get("touched"))
        out["distractor_touched"] = bool(distr.get("touched"))

    closed = first_close_step(ep.get("steps", []), threshold) is not None
    touched_any = out["instructed_touched"] or out["distractor_touched"]
    if not closed:
        out["never_closed"] = True
    elif touched_any:
        out["closed_on_something"] = True
    else:
        # gripper closed but neither condition object was ever touched
        out["closed_on_nothing"] = True
    return out


def summarise(folder, instructed_idx, distractor_idx, threshold):
    folder = pathlib.Path(folder)
    files = sorted(folder.glob("**/trajectories/*.json"))
    if not files:
        raise SystemExit(f"no trajectory JSONs found under {folder}")

    rows = []
    for fp in files:
        try:
            ep = json.loads(fp.read_text())
        except Exception as e:
            print(f"skip {fp}: {e}")
            continue
        if "per_condition" not in ep:
            continue  # standard (non-CF) suite, no instructed/distractor split
        rows.append(classify_episode(ep, instructed_idx, distractor_idx, threshold))

    n = len(rows)
    if n == 0:
        raise SystemExit(f"{folder}: no counterfactual episodes with per_condition")

    pct = lambda key: round(100.0 * sum(r[key] for r in rows) / n, 1)
    return {
        "folder": str(folder),
        "episodes": n,
        "instructed_placement_pct": pct("instructed_success"),
        "distractor_placement_pct": pct("distractor_success"),
        "first_grasp": {
            "instructed_pct": pct("instructed_touched"),
            "distractor_pct": pct("distractor_touched"),
            "nothing_pct": pct("closed_on_nothing"),
            "never_closed_pct": pct("never_closed"),
        },
    }


def print_human(s):
    print(f"\n{s['folder']}  ({s['episodes']} episodes)")
    print(f"  Instructed placement : {s['instructed_placement_pct']:5.1f}%")
    print(f"  Distractor placement : {s['distractor_placement_pct']:5.1f}%")
    fg = s["first_grasp"]
    print("  First grasp:")
    print(f"    instructed object  : {fg['instructed_pct']:5.1f}%")
    print(f"    distractor object  : {fg['distractor_pct']:5.1f}%")
    print(f"    nothing (closed, no object touched) : {fg['nothing_pct']:5.1f}%")
    print(f"    never closed       : {fg['never_closed_pct']:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="results folder(s), e.g. results/early_k4")
    ap.add_argument("--instructed-index", type=int, default=0,
                    help="index of the instructed condition (default 0)")
    ap.add_argument("--distractor-index", type=int, default=1,
                    help="index of the distractor condition (default 1)")
    ap.add_argument("--close-threshold", type=float, default=0.04,
                    help="gripper-opening threshold below which it counts as closed")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    results = [summarise(f, args.instructed_index, args.distractor_index, args.close_threshold)
               for f in args.folders]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for s in results:
            print_human(s)


if __name__ == "__main__":
    main()
