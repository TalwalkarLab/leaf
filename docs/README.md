# Building Documentation
This repo uses `sphinx-autodoc` for auto-documentation generation, and Sphinx ReadTheDocs theme for rendering documentation/tutorials.

## Setup
Execute `pip install -r doc-dec-requirements.txt`

## Adding to documentation
For adding pages to the website:

- Create the pages in reStructuredText (`.rst`) or in Markdown (`.md`) format
- Add the pages to the appropriate directory under `<root>/docs/source/`
- Place links in the appropriate index file (such as `index.rst` in the subfolder)
- Re-create documentation. If your file hasn't been linked correctly, Sphinx will probably generate a warning about the file not being present under any tree

## Re-create documentation
Execute `make html` within this (`<root>/docs/`) folder

