PROJECT_DIR=$(pwd)/..

cookiecutter https://github.com/nicain/pyproject_template -o ../.. --config-file .cookiecutter.yaml --no-input --overwrite-if-exists --checkout 25

# Add post-cookiecutter commands that you always want run here:
git checkout -- $PROJECT_DIR/README.md
git checkout -- $PROJECT_DIR/AUTHORS.rst
git checkout -- $PROJECT_DIR/requirements.txt

# Enter patch mode on remaining diffs:
git add -p