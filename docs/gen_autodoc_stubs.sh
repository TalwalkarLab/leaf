SOURCE_DIR="${1:?Need documentation source directory to generate docs (such as docs/source)}"
MODEL_DIR="${2:?Need source directory to run autodoc (such as ./models)}"

AUTODOC_DIR="${SOURCE_DIR}/autodoc/"
mkdir -p "${AUTODOC_DIR}"
rm -rf "${AUTODOC_DIR}/*"

echo "Generating autodoc stubs to ${SOURCE_DIR}"
sphinx-apidoc -o "${AUTODOC_DIR}" "${MODEL_DIR}"
