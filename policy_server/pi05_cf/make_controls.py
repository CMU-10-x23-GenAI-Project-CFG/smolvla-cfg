import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent
BENCH = HERE.parent.parent / "third_party" / "libero-cf" / "libero" / "libero"


def name(token):
    return re.sub(r"_\d+$", "", token).replace("_", " ")


def prompt_of(bddl):
    return re.sub(r"^\d+-|\.bddl$", "", bddl.name).replace("_", " ")


def training_instruction(obj):
    return f"pick up the {obj} and place it in the basket"


def main():
    conditions = json.loads((BENCH / "conditions" / "libero_cf_object.json").read_text())
    trained = {prompt_of(p) for p in (BENCH / "bddl_files" / "libero_object").glob("*.bddl")}
    controls = {}
    for key, conds in sorted(conditions.items()):
        prompt = prompt_of(pathlib.Path(key))
        instructed, distractor = (name(c.split()[1]) for c in conds)
        assert prompt == training_instruction(instructed), (prompt, instructed)
        bddl = (BENCH / "bddl_files" / key).read_text()
        block = re.search(r"\(:objects(.*?)\)", bddl, re.S).group(1)
        scene = [name(line.split("-")[0].strip()) for line in block.strip().splitlines()]
        third = [o for o in scene if o not in ("basket", instructed, distractor) and training_instruction(o) in trained]
        controls[prompt] = {
            "instructed": instructed,
            "distractor": distractor,
            "negative": training_instruction(distractor),
            "empty": "",
            "wrong": training_instruction(third[0]),
            "wrong_object": third[0],
            "paraphrase": f"move the {distractor} into the basket",
            "generic": "pick up the object and place it in the basket",
        }
    (HERE / "controls.json").write_text(json.dumps(controls, indent=1) + "\n")


if __name__ == "__main__":
    main()
