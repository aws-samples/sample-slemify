# marked

The UI renders the agent's markdown answers with marked. The orchestrator image
ships the file, so the UI loads nothing from the internet.

- Version: 18.2.1
- Tarball: https://registry.npmjs.org/marked/-/marked-18.2.1.tgz
- Integrity: `sha512-ii4bxCTCqqwpT86XZWgJNs+ZnAIZksmLGYMViWA7PDIXYIeNo1x6cDdX3JOip9NM/cI6p8z57FxliACboTruMA==`
- Files: `marked.umd.js` is `package/lib/marked.umd.js` and `LICENSE` is
  `package/LICENSE` from the tarball, both unmodified.
- License: MIT (see `LICENSE`).

`server.py` serves `marked.umd.js` at `/vendor/marked/marked.umd.js`, and
`ui.html` loads it with the relative path `vendor/marked/marked.umd.js`. The
path must stay relative because the IDE serves the UI under `/ports/8000/`.

To update, download the new tarball, check its integrity against the npm
registry entry, and replace both files.
