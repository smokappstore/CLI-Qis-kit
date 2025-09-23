<div align="center">

![Qiskit CLI Logo](https://github.com/user-attachments/assets/e162a5f0-54e3-4845-ae4a-896aa08da3f9)


</div>

<img width="1100" height="506" alt="cliqis" src="https://github.com/user-attachments/assets/24b4ec52-5678-4a64-a673-4de48eb5a5fa" />

<div align="center">

              Qiskit Runtime CLI Qis-kit v2.1 - Constructor de Circuitos Interactivos
              🚀 Herramienta de línea de comandos interactiva para computación cuántica
 
              Desarrollado por SmokAppSoftware jako con Claude AI, Gemini AI, y GitHub Copilot
</div>

![Quantum Badge](https://img.shields.io/badge/quantum-system-blue)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Qis-kit CLI](https://img.shields.io/badge/Qiskit-runtime-yellow)


## 🌟 Características Principales


- ✅ **Constructor de circuitos interactivo** - Crea circuitos cuánticos paso a paso
- ✅ **Simulador local** - Funciona sin conexión a internet
- ✅ **Hardware cuántico real** - Conexión a IBM Quantum (opcional)
- ✅ **Demostraciones educativas** - Aprende con ejemplos interactivos
- ✅ **Visualizaciones** - Gráficos y histogramas automáticos
- ✅ **Interfaz colorizada** - Experiencia visual mejorada
- ✅ **Instalación automática** - Setup guiado y verificación de dependencias


## 🚀 Instalación Rápida

### Opción 1: Instalación Automática (Recomendada)

```bash
# Clona el repositorio
git clone https://github.com/smokapp/qiskit-runtime-cli.git
cd qiskit-runtime-cli

# Ejecuta el instalador automático
python install.py
```

### Opción 2: Instalación Manual

```bash
# Instala las dependencias
pip install -r requirements.txt

# Ejecuta la CLI
python main.py
```

### Opción 3: Usando Make

```bash
# Ve todos los comandos disponibles
make help

# Instalación completa
make install

# Ejecutar
make run
```

## 📦 Requisitos del Sistema

- **Python 3.8 o superior**
- **Conexión a internet** (solo para instalar dependencias e IBM Quantum)
- **Sistema operativo**: Windows, macOS, Linux

### Dependencias Principales

```bash
qiskit>=0.45.0          # Framework cuántico
qiskit-aer>=0.12.0      # Simulador local
matplotlib>=3.5.0       # Visualizaciones
numpy>=1.21.0           # Operaciones numéricas
```

### Dependencias Opcionales

```bash
qiskit-ibm-runtime>=0.15.0  # Para hardware cuántico real
```

## 🎯 Inicio Rápido

### 1. Primera Ejecución

```bash
python main.py
```

La CLI detectará automáticamente qué dependencias tienes instaladas y te guiará.

### 2. Tu Primer Circuito Cuántico

```
(circuito: vacío) » crear 2          # Crea un circuito de 2 qubits
(circuito: 2Q) » agregar h 0         # Aplica Hadamard al qubit 0
(circuito: 2Q) » agregar cx 0 1      # Aplica CNOT entre qubits 0 y 1
(circuito: 2Q) » ver                 # Ve tu circuito
(circuito: 2Q) » ejecutar            # Ejecuta en el simulador local
```

### 3. Explorar Demostraciones

```
(circuito: vacío) » demo
```

Esto ejecutará una serie de demostraciones educativas que te enseñarán:
- Estados de superposición
- Entrelazamiento cuántico
- Puertas de Pauli (X, Y, Z)
- Puertas de fase (S, T)
- Rotaciones cuánticas
- Circuitos complejos

## 🔧 Comandos Disponibles

### Constructor de Circuitos

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `crear <qubits>` | Inicia un nuevo circuito | `crear 3` |
| `agregar <puerta> <args...>` | Añade una puerta | `agregar h 0` |
| `ver` | Muestra el circuito actual | `ver` |
| `medir [all\|qubits...]` | Añade mediciones | `medir all` |
| `ejecutar [backend] [shots]` | Ejecuta el circuito | `ejecutar` |
| `limpiar` | Elimina el circuito actual | `limpiar` |

### Puertas Cuánticas Soportadas

#### Puertas de 1 Qubit
- `h <qubit>` - Hadamard (superposición)
- `x <qubit>` - Pauli-X (NOT cuántico)
- `y <qubit>` - Pauli-Y
- `z <qubit>` - Pauli-Z (cambio de fase)
- `s <qubit>` - Fase π/2
- `t <qubit>` - Fase π/4
- `rx <ángulo> <qubit>` - Rotación X
- `ry <ángulo> <qubit>` - Rotación Y
- `rz <ángulo> <qubit>` - Rotación Z

#### Puertas de 2 Qubits
- `cx <control> <target>` - CNOT
- `cz <control> <target>` - Controlled-Z
- `swap <q1> <q2>` - Intercambio
- `crz <ángulo> <control> <target>` - Controlled-RZ

#### Puertas de 3 Qubits
- `ccx <c1> <c2> <target>` - Toffoli
- `cswap <control> <t1> <t2>` - Fredkin

### Ejemplos de Ángulos

```bash
agregar rx pi 0        # π radianes
agregar ry pi/2 1      # π/2 radianes
agregar rz pi/4 2      # π/4 radianes
agregar rx 1.5708 0    # 1.5708 radianes (≈ π/2)
```

## 🖥️ Modos de Ejecución

### 1. Simulador Local (Por defecto)
- ✅ Funciona sin internet
- ✅ Ilimitados qubits (limitado por RAM)
- ✅ Ejecución instantánea
- ✅ Perfecto para aprendizaje

### 2. IBM Quantum (Opcional)
- 🌐 Requiere cuenta de IBM Quantum
- ⚛️ Hardware cuántico real
- 📊 Cola de trabajos
- 🎯 Resultados de computadoras cuánticas reales

#### Configuración IBM Quantum

1. **Regístrate en IBM Quantum**
   ```
   https://quantum.ibm.com/
   ```

2. **Obtén tu API Token**
   - Ve a Account Settings
   - Copia tu API Token

3. **Configura en la CLI**
   ```
   (circuito: vacío) » configurar
   🔑 Introduce tu token de API: [pega_tu_token_aquí]
   ```

4. **Prueba la conexión**
   ```
   (circuito: vacío) » test
   ```

## 📊 Ejemplos Completos

### Circuito de Bell (Entrelazamiento)

```bash
# Crear circuito de 2 qubits
crear 2

# Crear superposición en qubit 0
agregar h 0

# Entrelazar qubits 0 y 1
agregar cx 0 1

# Ver el circuito
ver

# Ejecutar 1000 veces
ejecutar local_simulator 1000
```

**Resultado esperado**: 50% |00⟩, 50% |11⟩

### Circuito de Rotación Cuántica

```bash
# Crear circuito de 1 qubit
crear 1

# Aplicar rotaciones
agregar rx pi/2 0
agregar ry pi/4 0
agregar rz pi/8 0

# Ver y ejecutar
ver
ejecutar
```

### Circuito Complejo (3 Qubits)

```bash
crear 3
agregar h 0
agregar h 1
agregar h 2
agregar ccx 0 1 2    # Toffoli gate
agregar swap 1 2
ejecutar
```

## 🎓 Recursos de Aprendizaje

### Comando Demo
El comando `demo` incluye tutoriales interactivos sobre:

1. **Superposición**: Puerta Hadamard y probabilidades
2. **Entrelazamiento**: Estados de Bell y correlaciones
3. **Puertas de Pauli**: X, Y, Z y sus efectos
4. **Puertas de Fase**: S, T y interferencia cuántica
5. **Rotaciones**: Control preciso del estado cuántico
6. **Circuitos Complejos**: Combinación de múltiples técnicas

### Comandos de Información
- `ayuda` - Lista completa de comandos
- `backends` - Ve backends disponibles
- `agregar` (sin argumentos) - Lista de puertas disponibles

## 🛠️ Desarrollo y Contribución

### Setup de Desarrollo

```bash
# Clona el repositorio
git clone https://github.com/smokapp/qiskit-runtime-cli.git
cd qiskit-runtime-cli

# Configura entorno de desarrollo
make install-dev

# Ejecuta tests
make test

# Formatea código
make format

# Verifica código
make lint
```

### Estructura del Proyecto

```
qiskit-runtime-cli/
├── main.py              # Punto de entrada inteligente
├── qiskit_cli.py        # CLI original (IBM only)
├── run_cli.py           # CLI completa (local + IBM)
├── install.py           # Instalador automático
├── requirements.txt     # Dependencias
├── setup.py            # Configuración de instalación
├── pyproject.toml      # Configuración moderna
├── Makefile            # Automatización de tareas
└── README.md           # Esta documentación
```

### Arquitectura

- **main.py**: Detecta dependencias y ejecuta la versión apropiada
- **run_cli.py**: Versión completa con simulador local e IBM Quantum
- **qiskit_cli.py**: Versión original optimizada para IBM Quantum
- **install.py**: Instalador interactivo con verificación de dependencias

## 🔧 Comandos Make Disponibles

```bash
make help              # Ver todos los comandos
make install           # Instalación completa
make install-minimal   # Instalación solo simulador local
make run              # Ejecutar CLI
make test             # Ejecutar pruebas
make clean            # Limpiar archivos temporales
make format           # Formatear código
make lint             # Verificar código
make check-deps       # Verificar dependencias
make info             # Información del proyecto
make status           # Estado completo del proyecto
```

## ❓ Preguntas Frecuentes

### ¿Puedo usar la CLI sin internet?
✅ **Sí**, el simulador local funciona completamente offline después de la instalación.

### ¿Necesito una cuenta de IBM Quantum?
**No es obligatorio**. La CLI funciona perfectamente con el simulador local. IBM Quantum es opcional para usar hardware real.

### ¿Cuántos qubits puedo simular?
El límite depende de tu RAM. Generalmente:
- **20+ qubits**: Simulación rápida
- **30+ qubits**: Simulación lenta pero posible
- **40+ qubits**: Requiere mucha RAM

### ¿Funciona en Windows/Mac/Linux?
✅ **Sí**, es compatible con todos los sistemas operativos principales.

### ¿Puedo contribuir al proyecto?
✅ **¡Por supuesto!** Las contribuciones son bienvenidas. Ve la sección de desarrollo arriba.

## 📄 Licencia

Este proyecto está licenciado bajo la **Apache License 2.0**. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **IBM Quantum** - Por proporcionar acceso a hardware cuántico real
- **Qiskit Team** - Por el excelente framework de computación cuántica
- **Claude AI, Gemini AI, GitHub Copilot** - Por asistencia en el desarrollo
- **Comunidad Cuántica** - Por feedback y sugerencias

## 📞 Soporte

- **Issues**: [GitHub Issues](https://github.com/smokappstore/CLI-Qis-kit-/issues)
- **Email**: jakocrazykings@gmail.com
- **Documentación**: Este README.md

---

<div align="center">

**🌟 ¡Dale una estrella si este proyecto te ayuda! 🌟**

*¡Hecho con ❤️ para la comunidad de computación cuántica!*

</div>
