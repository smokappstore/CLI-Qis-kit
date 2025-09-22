# Makefile para Qiskit Runtime CLI Qis-kit
# by SmokAppSoftware jako with Claude AI, Gemini AI, COPILOT
# Versión 2.2

.PHONY: help install install-dev install-minimal run clean test format lint docs

# Colores para output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT_NAME := qiskit-runtime-cli

help: ## Muestra esta ayuda
	@echo "$(GREEN)Qiskit Runtime CLI - Comandos disponibles:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Instalación completa (simulador + IBM Quantum)
	@echo "$(GREEN)🚀 Instalando Qiskit Runtime CLI (completo)...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Instalación completada$(NC)"

install-dev: ## Instalación para desarrollo (incluye herramientas de testing)
	@echo "$(GREEN)🛠️  Instalando para desarrollo...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy
	@echo "$(GREEN)✅ Instalación de desarrollo completada$(NC)"

install-minimal: ## Instalación mínima (solo simulador local)
	@echo "$(GREEN)📦 Instalación mínima...$(NC)"
	$(PIP) install qiskit qiskit-aer matplotlib numpy Pillow
	@echo "$(GREEN)✅ Instalación mínima completada$(NC)"

install-setup: ## Instala usando setup.py
	@echo "$(GREEN)📦 Instalando con setup.py...$(NC)"
	$(PYTHON) setup.py install
	@echo "$(GREEN)✅ Instalación con setup.py completada$(NC)"

run: ## Ejecuta la CLI principal
	@echo "$(GREEN)🚀 Iniciando Qiskit Runtime CLI...$(NC)"
	$(PYTHON) main.py

run-installer: ## Ejecuta el instalador automático
	@echo "$(GREEN)🔧 Ejecutando instalador automático...$(NC)"
	$(PYTHON) install.py

run-original: ## Ejecuta la CLI original (solo IBM)
	@echo "$(GREEN)☁️  Iniciando CLI original (IBM Quantum)...$(NC)"
	$(PYTHON) qiskit_cli.py

run-local: ## Ejecuta la CLI con simulador local
	@echo "$(GREEN)🖥️  Iniciando CLI con simulador local...$(NC)"
	$(PYTHON) run_cli.py

clean: ## Limpia archivos generados
	@echo "$(YELLOW)🧹 Limpiando archivos generados...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	rm -f qiskit-cli.py
	@echo "$(GREEN)✅ Limpieza completada$(NC)"

test: ## Ejecuta las pruebas
	@echo "$(GREEN)🧪 Ejecutando pruebas...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=term-missing
	@echo "$(GREEN)✅ Pruebas completadas$(NC)"

test-simple: ## Prueba simple de funcionamiento
	@echo "$(GREEN)🔍 Verificando funcionamiento básico...$(NC)"
	$(PYTHON) -c "import main; print('✅ main.py importa correctamente')"
	$(PYTHON) -c "import qiskit_cli; print('✅ qiskit_cli.py importa correctamente')" 
	$(PYTHON) -c "import run_cli; print('✅ run_cli.py importa correctamente')"
	@echo "$(GREEN)✅ Verificación básica completada$(NC)"

format: ## Formatea el código con black
	@echo "$(GREEN)🎨 Formateando código...$(NC)"
	$(PYTHON) -m black *.py --line-length 100
	@echo "$(GREEN)✅ Formateo completado$(NC)"

lint: ## Verifica el código con flake8
	@echo "$(GREEN)🔍 Verificando código...$(NC)"
	$(PYTHON) -m flake8 *.py --max-line-length=100 --ignore=E203,W503
	@echo "$(GREEN)✅ Verificación completada$(NC)"

type-check: ## Verifica tipos con mypy
	@echo "$(GREEN)🔍 Verificando tipos...$(NC)"
	$(PYTHON) -m mypy *.py --ignore-missing-imports
	@echo "$(GREEN)✅ Verificación de tipos completada$(NC)"

check-deps: ## Verifica las dependencias instaladas
	@echo "$(GREEN)📋 Verificando dependencias...$(NC)"
	@$(PYTHON) -c "import qiskit; print('✅ Qiskit:', qiskit.__version__)" || echo "$(RED)❌ Qiskit no instalado$(NC)"
	@$(PYTHON) -c "import qiskit_aer; print('✅ Qiskit Aer:', qiskit_aer.__version__)" || echo "$(YELLOW)⚠️  Qiskit Aer no instalado$(NC)"
	@$(PYTHON) -c "import qiskit_ibm_runtime; print('✅ IBM Runtime:', qiskit_ibm_runtime.__version__)" || echo "$(YELLOW)⚠️  IBM Runtime no instalado$(NC)"
	@$(PYTHON) -c "import matplotlib; print('✅ Matplotlib:', matplotlib.__version__)" || echo "$(YELLOW)⚠️  Matplotlib no instalado$(NC)"
	@$(PYTHON) -c "import numpy; print('✅ NumPy:', numpy.__version__)" || echo "$(YELLOW)⚠️  NumPy no instalado$(NC)"

build: ## Construye el paquete para distribución
	@echo "$(GREEN)📦 Construyendo paquete...$(NC)"
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)✅ Paquete construido en dist/$(NC)"

upload-test: ## Sube el paquete a TestPyPI
	@echo "$(YELLOW)📤 Subiendo a TestPyPI...$(NC)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)✅ Subido a TestPyPI$(NC)"

upload: ## Sube el paquete a PyPI (producción)
	@echo "$(RED)📤 Subiendo a PyPI (PRODUCCIÓN)...$(NC)"
	$(PYTHON) -m twine upload dist/*
	@echo "$(GREEN)✅ Subido a PyPI$(NC)"

docs: ## Genera documentación
	@echo "$(GREEN)📚 Generando documentación...$(NC)"
	@echo "README.md ya existe como documentación principal"
	@echo "Para más documentación, considera usar Sphinx"
	@echo "$(GREEN)✅ Documentación lista$(NC)"

demo: ## Ejecuta la CLI en modo demo
	@echo "$(GREEN)🎯 Iniciando demo de Qiskit CLI...$(NC)"
	@echo "Ejecuta 'demo' dentro de la CLI para ver ejemplos"
	$(PYTHON) main.py

create-launcher: ## Crea un script launcher ejecutable
	@echo "$(GREEN)🔧 Creando launcher...$(NC)"
	@echo "#!/usr/bin/env python3" > qiskit-cli.py
	@echo "import sys" >> qiskit-cli.py
	@echo "from pathlib import Path" >> qiskit-cli.py
	@echo "sys.path.insert(0, str(Path(__file__).parent))" >> qiskit-cli.py
	@echo "import main" >> qiskit-cli.py
	@echo "main.main()" >> qiskit-cli.py
	@chmod +x qiskit-cli.py 2>/dev/null || true
	@echo "$(GREEN)✅ Launcher creado: qiskit-cli.py$(NC)"

info: ## Muestra información del proyecto
	@echo "$(GREEN)ℹ️  Información del proyecto:$(NC)"
	@echo ""
	@echo "Nombre: Qiskit Runtime CLI v2.2"
	@echo "Autor: SmokAppSoftware jako"
	@echo "Descripción: Constructor de circuitos cuánticos interactivo"
	@echo "Python: $(shell python3 --version)"
	@echo "Directorio: $(PWD)"
	@echo ""
	@echo "Archivos principales:"
	@echo "  - main.py           (punto de entrada)"
	@echo "  - qiskit_cli.py     (CLI original - IBM only)"
	@echo "  - run_cli.py        (CLI completa - local + IBM)"
	@echo "  - install.py        (instalador automático)"
	@echo ""

requirements-freeze: ## Genera requirements.txt actualizado
	@echo "$(GREEN)📋 Generando requirements.txt actualizado...$(NC)"
	$(PIP) freeze > requirements-frozen.txt
	@echo "$(GREEN)✅ requirements-frozen.txt generado$(NC)"

venv: ## Crea un entorno virtual
	@echo "$(GREEN)🐍 Creando entorno virtual...$(NC)"
	$(PYTHON) -m venv venv
	@echo "$(GREEN)✅ Entorno virtual creado en ./venv$(NC)"
	@echo "$(YELLOW)Actívalo con: source venv/bin/activate (Linux/Mac) o venv\\Scripts\\activate (Windows)$(NC)"

venv-install: venv ## Crea entorno virtual e instala dependencias
	@echo "$(GREEN)📦 Instalando en entorno virtual...$(NC)"
	./venv/bin/python -m pip install --upgrade pip
	./venv/bin/python -m pip install -r requirements.txt
	@echo "$(GREEN)✅ Entorno virtual configurado$(NC)"

# Comandos de desarrollo rápido
dev: install-dev format lint test-simple ## Setup completo de desarrollo

quick-test: ## Test rápido sin cobertura
	@echo "$(GREEN)⚡ Test rápido...$(NC)"
	$(PYTHON) main.py --help 2>/dev/null || echo "CLI requiere modo interactivo"
	@echo "$(GREEN)✅ Test rápido completado$(NC)"

# Comandos de debugging
debug-main: ## Debug del main.py
	@echo "$(GREEN)🐛 Debug main.py...$(NC)"
	$(PYTHON) -c "import main; print('Dependencies check:'); main.check_dependencies()"

debug-imports: ## Verifica todas las importaciones
	@echo "$(GREEN)🔍 Verificando importaciones...$(NC)"
	@$(PYTHON) -c "import sys; print('Python path:', sys.path[0])"
	@$(PYTHON) -c "import main; print('✅ main.py OK')" || echo "$(RED)❌ Error en main.py$(NC)"
	@$(PYTHON) -c "import qiskit_cli; print('✅ qiskit_cli.py OK')" || echo "$(RED)❌ Error en qiskit_cli.py$(NC)"
	@$(PYTHON) -c "import run_cli; print('✅ run_cli.py OK')" || echo "$(RED)❌ Error en run_cli.py$(NC)"

# Comandos de limpieza específicos
clean-logs: ## Limpia logs generados
	@echo "$(YELLOW)🗑️  Limpiando logs...$(NC)"
	rm -rf ~/.qiskit_cli/logs/*
	@echo "$(GREEN)✅ Logs limpiados$(NC)"

clean-cache: ## Limpia cache de Python
	@echo "$(YELLOW)🗑️  Limpiando cache...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cache limpiado$(NC)"

# Meta comando que muestra el estado del proyecto
status: ## Muestra el estado completo del proyecto
	@echo "$(GREEN)📊 Estado del Proyecto Qiskit Runtime CLI$(NC)"
	@echo "=================================================="
	@make info
	@echo ""
	@make check-deps
	@echo ""
	@echo "$(GREEN)📁 Archivos del proyecto:$(NC)"
	@ls -la *.py 2>/dev/null || echo "No hay archivos Python"
	@echo ""
	@echo "$(GREEN)📝 Archivos de configuración:$(NC)"
	@ls -la requirements.txt setup.py Makefile 2>/dev/null || echo "Archivos de config no encontrados"

# Comando por defecto
all: install test ## Instalación completa + tests
