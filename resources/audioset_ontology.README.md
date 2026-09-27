# Official AudioSet ontology

`audioset_ontology.json` is an unmodified snapshot downloaded on 2026-09-27
from https://raw.githubusercontent.com/audioset/ontology/master/ontology.json.

- Publisher: Google Inc., AudioSet project, Dan Ellis.
- Official website: https://research.google.com/audioset/ontology/index.html
- Repository and schema: https://github.com/audioset/ontology
- License: Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0), https://creativecommons.org/licenses/by-sa/4.0/
- SHA-256: `9c685f4403eecc3ca9be37fd7285cf212feaaea6ff7229d3e7ca89e0d1f2d15d`
- Nodes: 632. The full graph includes nodes outside the model's 527-class map;
  keep these intermediate nodes when resolving ancestor/descendant paths.

`child_ids` defines directed parent-to-child edges. A node may have more than
one parent. Diagnostic reports use this graph only to describe relationships;
they do not propagate labels, change masks or modify training supervision.
