# ====== User configuration ======
dir = Tilt-Energy-Technical-Case-Electricity-Consumption
VENV = venv
SYS_PY ?= python  # fallback system Python (use python3 if needed)

# ====== Platform detection ======
ifeq ($(OS),Windows_NT)
  VENV_BIN := $(VENV)/Scripts
  PYTHON   := $(VENV_BIN)/python.exe
  ACTIVATE := .\$(VENV_BIN)\Activate.ps1
else
  VENV_BIN := $(VENV)/bin
  PYTHON   := $(VENV_BIN)/python
  ACTIVATE := source $(VENV_BIN)/activate
endif

# ====== Targets ======
.PHONY: add-venv install init setup jupyter-venv-add jupyter-venv-remove clean-venv show-activate

# Create a new virtual environment
add-venv:
	$(SYS_PY) -m venv $(VENV)

# Clean the venv in a portable way (Python handles deletion)
clean-venv:
	$(SYS_PY) -c "import shutil, sys; shutil.rmtree('$(VENV)', ignore_errors=True)"

# Install dependencies into the venv (no manual activation needed)
install: $(VENV_BIN)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Add the venv as a Jupyter kernel
jupyter-venv-add: $(VENV_BIN)
	$(PYTHON) -m ipykernel install --name=$(dir) --user

# Remove the Jupyter kernel (the dash ignores error if it does not exist)
jupyter-venv-remove: $(VENV_BIN)
	-$(PYTHON) -m jupyter kernelspec uninstall $(dir) -y

# Initialize everything (venv + fresh Jupyter kernel)
init: clean-venv add-venv 

# Full setup (init + dependency install)
setup: init install jupyter-venv-remove jupyter-venv-add



# Show the correct activation command depending on OS
show-activate:
	@echo "To activate the virtual environment:"
ifeq ($(OS),Windows_NT)
	@echo "  PowerShell : $(ACTIVATE)"
	@echo "  cmd.exe    : $(VENV)\\Scripts\\activate.bat"
else
	@echo "  bash/zsh   : $(ACTIVATE)"
endif


format:
	$(PYTHON) -m black src
	$(PYTHON) -m isort src --settings-file ./setup.cfg

format-check:
	$(PYTHON) -m black src --check
	$(PYTHON) -m isort src --check-only --settings-file ./setup.cfg

lint-check:
	$(PYTHON) -m pylint src --rcfile ./setup.cfg
	$(PYTHON) -m flake8 src --config ./setup.cfg