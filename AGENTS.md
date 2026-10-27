# Asset versions

When changing `app.js`, `style.css`, or their references in `index.html`:

- After all edits, run `python3 scripts/update-asset-versions.py`.
- Run `python3 scripts/update-asset-versions.py --check` before finishing.
- Include the resulting `index.html` changes with the asset changes.
- Keep the content-based `?v=` parameters on script and stylesheet URLs.
