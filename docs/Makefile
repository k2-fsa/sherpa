# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# List all scripts to run before HTML build, with their paths
SCRIPTS = \
    source/onnx/install/windows/run.sh \
    source/onnx/install/windows/run.sh



# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile prebuild

prebuild:
	@echo "Running prebuild scripts..."
	@for script in $(SCRIPTS); do \
	    dir=$$(dirname $$script); \
	    file=$$(basename $$script); \
	    echo "Running $$file in $$dir..."; \
			cd $$dir && echo "PWD: $$PWD" ./$$file && cd -; \
	done

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@if [ "$@" = "html" ]; then \
	    $(MAKE) prebuild; \
	fi
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
