#Dockerfile vars

#vars

.PHONY: help build bootstrap all

help:
	    @echo "Makefile arguments:"
	    @echo ""
	    @echo "Makefile commands:"
	    @echo "build"
	    @echo "all"
			@echo "publish"
			@echo ${TAG}

.DEFAULT_GOAL := all

build:	
	@echo ">>>> Build python module"
	@python3 setup.py sdist bdist_wheel	

upload:
	@python3 -m twine upload --repository pypi dist/*

install:	
	@echo ">>>> Install python module"
	@pip3 install .

install-dev:	
	@echo ">>>> Install python module development"
	@pip3 install -e .

docs:
	@echo ">>>> Build docs"
	$(MAKE) -C $@

all: build
