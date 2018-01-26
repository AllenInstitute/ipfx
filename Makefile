
REQUIREMENTS := $(wildcard requirements*.txt)

clean: clean-requirements
	make -C docs clean
	rm .tested_requirements.txt

clean-requirements: $(REQUIREMENTS)
	rm -f $(REQUIREMENTS)

requirements:
	pip-compile requirements.in
	pip-compile requirements-doc.in
	pip-compile requirements-test.in
	pip-compile requirements-dev.in
