mkdir -p autodoc/
rm -rf autodoc/*
sphinx-apidoc -o autodoc/ ../../models
