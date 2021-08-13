sphinx-apidoc -f -o ./docs/source/ ./
#sphinx-apidoc --ext-autodoc -f -o ./docs/source/ ./
cd docs
make html
