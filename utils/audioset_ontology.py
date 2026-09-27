"""Read the full official AudioSet DAG, including non-model intermediate nodes."""

import json
from collections import deque
from pathlib import Path


class AudioSetOntology:
    def __init__(self, records):
        self.nodes = {row["id"]: row for row in records}
        if not self.nodes or len(self.nodes) != len(records):
            raise ValueError("Ontology must contain unique, nonempty node IDs")
        self.parents = {mid: set() for mid in self.nodes}
        for mid, row in self.nodes.items():
            for child in row["child_ids"]:
                if child not in self.nodes:
                    raise ValueError(f"Unknown ontology child: {child}")
                self.parents[child].add(mid)
        # Validate the DAG without assuming a single parent or a fixed depth.
        indegree = {mid: len(parents) for mid, parents in self.parents.items()}
        queue = deque(mid for mid, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            mid = queue.popleft()
            visited += 1
            for child in set(self.nodes[mid]["child_ids"]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != len(self.nodes):
            raise ValueError("Ontology contains a cycle")
        self._paths = {}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def descendant_paths(self, start):
        if start not in self._paths:
            paths = {start: [start]}
            queue = deque([start])
            while queue:
                mid = queue.popleft()
                for child in sorted(self.nodes[mid]["child_ids"]):
                    if child not in paths:
                        paths[child] = paths[mid] + [child]
                        queue.append(child)
            self._paths[start] = paths
        return self._paths[start]

    def relation(self, target, other):
        result = {"ontology_relation": "unknown_mid", "ontology_distance": None,
                  "ontology_path": "", "is_ancestor_or_descendant": None,
                  "shared_parent_mids": ""}
        if target not in self.nodes or other not in self.nodes:
            return result
        result.update(ontology_relation="no_lineage_relation", is_ancestor_or_descendant=False,
                      shared_parent_mids=";".join(sorted(self.parents[target] & self.parents[other])))
        if target == other:
            return {**result, "ontology_relation": "same", "ontology_distance": 0,
                    "ontology_path": target}
        path = self.descendant_paths(target).get(other)
        if path:
            relation = "other_is_child" if len(path) == 2 else "other_is_descendant"
        else:
            path = self.descendant_paths(other).get(target)
            relation = "other_is_parent" if path and len(path) == 2 else "other_is_ancestor"
        if path:
            result.update(ontology_relation=relation, ontology_distance=len(path) - 1,
                          ontology_path=" > ".join(path), is_ancestor_or_descendant=True)
        return result
