
<img width="280" height="306" alt="cliqis" src="https://github.com/user-attachments/assets/24b4ec52-5678-4a64-a673-4de48eb5a5fa" />


# Qiskit Runtime CLI v2.1 - Constructor de Circuitos Interactivos
src= "https://img.shields.io/badge/quantum-system-blue" 
<img width="650" height="344" alt="logo" src="https://github.com/user-attachments/assets/e162a5f0-54e3-4845-ae4a-896aa08da3f9" />

**Desarrollado por:** SmokAppSoftware jako con Claude AI, Gemini AI, y COPILOT

## Descripción General

Qiskit Runtime CLI es una herramienta de línea de comandos interactiva que facilita la gestión de servicios de Qiskit Runtime de IBM Quantum. La versión 2.1 introduce un **Constructor de Circuitos Interactivo** que permite crear, modificar y ejecutar circuitos cuánticos de manera intuitiva desde la terminal.

### Características Principales

- ✅ **Gestión de cuenta IBM Quantum** - Configuración y validación de credenciales
- ✅ **Exploración de backends** - Lista y análisis de computadoras cuánticas disponibles
- ✅ **Constructor de circuitos interactivo** - Creación paso a paso de circuitos cuánticos
- ✅ **Ejecución en tiempo real** - Envío de trabajos y monitoreo de resultados
- ✅ **Logging avanzado** - Sistema completo de registros y depuración
- ✅ **Interfaz colorizada** - Experiencia visual mejorada en terminal

## Instalación y Requisitos

### Dependencias

```bash
pip install qiskit qiskit-ibm-runtime
```

### Ejecución

```bash
python qiskit_cli.py
```

## Guía de Uso

### 1. Configuración Inicial

Al ejecutar la CLI por primera vez, configura tu cuenta de IBM Quantum:

```
Qiskit-CLI > configurar
🔑 Introduce tu token de API de IBM Quantum: [tu_token_aquí]
```

**Obtener tu token:**
1. Ve a [IBM Quantum Dashboard](https://quantum.ibm.com/)
2. Navega a "Account Settings"
3. Copia tu API Token

### 2. Verificación de Conexión

```
Qiskit-CLI > probar
📡 Probando conexión con IBM Quantum...
✅ ¡Conexión exitosa en 1.23s! Encontrados 15 backends.
```

### 3. Exploración de Backends

#### Listar todos los backends operacionales
```
Qiskit-CLI > listar
```

#### Listar solo simuladores
```
Qiskit-CLI > listar --simuladores
```

#### Encontrar el backend menos ocupado
```
Qiskit-CLI > mejor
🎯 Backend menos ocupado: ibm_brisbane (Cola: 5 trabajos)
```

## Constructor de Circuitos Interactivo

### Estado del Sistema

La CLI mantiene en memoria el circuito actual mediante:
- `self.current_circuit: Optional[QuantumCircuit]` - El circuito en construcción

### Flujo de Trabajo

#### 1. Crear un Nuevo Circuito

```
Qiskit-CLI > crear 3
✅ Circuito de 3 qubits creado. ¡Listo para agregar puertas!
```

#### 2. Agregar Puertas Cuánticas

##### Puertas de Un Qubit
```
Qiskit-CLI > agregar h 0          # Puerta Hadamard en qubit 0
Qiskit-CLI > agregar x 1          # Puerta Pauli-X en qubit 1
Qiskit-CLI > agregar y 2          # Puerta Pauli-Y en qubit 2
Qiskit-CLI > agregar z 0          # Puerta Pauli-Z en qubit 0
Qiskit-CLI > agregar s 1          # Puerta S en qubit 1
Qiskit-CLI > agregar t 2          # Puerta T en qubit 2
```

##### Puertas de Dos Qubits
```
Qiskit-CLI > agregar cx 0 1       # CNOT: control=0, target=1
Qiskit-CLI > agregar cz 0 2       # Controlled-Z: control=0, target=2
Qiskit-CLI > agregar swap 1 2     # Intercambiar estados de qubits 1 y 2
```

##### Puertas de Rotación (con parámetros)
```
Qiskit-CLI > agregar rx pi/2 0    # Rotación X con ángulo π/2 en qubit 0
Qiskit-CLI > agregar ry pi/4 1    # Rotación Y con ángulo π/4 en qubit 1
Qiskit-CLI > agregar rz pi 2      # Rotación Z con ángulo π en qubit 2
Qiskit-CLI > agregar p pi/3 0     # Puerta de fase con ángulo π/3 en qubit 0
```

#### 3. Visualizar el Circuito

```
Qiskit-CLI > ver
--- Circuito Actual ---
     ┌───┐     ┌─────────┐
q_0: ┤ H ├──■──┤ Rx(π/2) ├
     └───┘┌─┴─┐└─────────┘
q_1: ─────┤ X ├───────────
          └───┘
q_2: ─────────────────────
```

#### 4. Ejecutar el Circuito

```
Qiskit-CLI > ejecutar_custom
🔧 Ejecutando tu circuito personalizado:
🎯 Usando backend: ibm_brisbane
⚙️ Preparando ejecución con 1024 shots...
🚀 Job enviado con ID: abc123-def456-ghi789
```

#### 5. Gestión del Circuito

##### Limpiar el circuito actual
```
Qiskit-CLI > limpiar_circuito
🗑️ Circuito actual eliminado.
```

## Puertas Cuánticas Soportadas

### Referencia Completa

| Puerta | Sintaxis | Descripción | Qubits | Parámetros |
|--------|----------|-------------|---------|------------|
| **H** | `agregar h <qubit>` | Hadamard | 1 | 0 |
| **X** | `agregar x <qubit>` | Pauli-X (NOT) | 1 | 0 |
| **Y** | `agregar y <qubit>` | Pauli-Y | 1 | 0 |
| **Z** | `agregar z <qubit>` | Pauli-Z | 1 | 0 |
| **S** | `agregar s <qubit>` | Fase π/2 | 1 | 0 |
| **S†** | `agregar sdg <qubit>` | Fase -π/2 | 1 | 0 |
| **T** | `agregar t <qubit>` | Fase π/4 | 1 | 0 |
| **T†** | `agregar tdg <qubit>` | Fase -π/4 | 1 | 0 |
| **CNOT** | `agregar cx <ctrl> <tgt>` | Controlled-X | 2 | 0 |
| **CZ** | `agregar cz <ctrl> <tgt>` | Controlled-Z | 2 | 0 |
| **SWAP** | `agregar swap <q1> <q2>` | Intercambio | 2 | 0 |
| **RX** | `agregar rx <ángulo> <qubit>` | Rotación X | 1 | 1 |
| **RY** | `agregar ry <ángulo> <qubit>` | Rotación Y | 1 | 1 |
| **RZ** | `agregar rz <ángulo> <qubit>` | Rotación Z | 1 | 1 |
| **P** | `agregar p <ángulo> <qubit>` | Fase arbitraria | 1 | 1 |

### Notación de Parámetros

La CLI soporta expresiones matemáticas para los ángulos:
- `pi` → π (3.14159...)
- `pi/2` → π/2
- `pi/4` → π/4
- `2*pi` → 2π
- `0.5` → 0.5 radianes

## Comandos de Gestión

### Monitoreo de Trabajos

```
Qiskit-CLI > estado abc123-def456-ghi789
--- Estado del Job ---
📋 ID: abc123-def456-ghi789
📊 Estado: RUNNING - Job is running
🖥️ Backend: ibm_brisbane
🔢 Posición en cola: 3
📅 Fecha de creación: 2025-01-15 14:30:22 UTC
```

### Ejecutar Circuito de Ejemplo

```
Qiskit-CLI > ejemplo
🔧 Circuito de Bell a ejecutar:
     ┌───┐     
q_0: ┤ H ├──■──
     └───┘┌─┴─┐
q_1: ─────┤ X ├
          └───┘
```

## Arquitectura del Sistema

### Componentes Principales

#### Clase `QiskitCLI`

**Atributos de Estado:**
- `service: Optional[QiskitRuntimeService]` - Conexión a IBM Quantum
- `current_circuit: Optional[QuantumCircuit]` - Circuito en construcción
- `logger: logging.Logger` - Sistema de logging

**Métodos del Constructor de Circuitos:**

##### `create_circuit(num_qubits: str)`
- **Propósito:** Inicializa un nuevo circuito cuántico
- **Validación:** Verifica que `num_qubits` sea un entero positivo
- **Estado:** Establece `self.current_circuit = QuantumCircuit(num_qubits)`

##### `add_gate_to_circuit(args: List[str])`
- **Propósito:** Añade puertas cuánticas al circuito actual
- **Procesamiento:**
  1. Parsea el nombre de la puerta y argumentos
  2. Valida contra `supported_gates` dictionary
  3. Convierte índices de qubits a enteros
  4. Evalúa parámetros matemáticos usando `eval()`
  5. Ejecuta el método correspondiente de Qiskit

##### `view_circuit()`
- **Propósito:** Visualiza el circuito usando ASCII art
- **Implementación:** Utiliza `qc.draw(output='text')`

##### `clear_circuit()`
- **Propósito:** Resetea el circuito actual
- **Acción:** `self.current_circuit = None`

##### `run_custom_circuit(backend_name, shots)`
- **Propósito:** Ejecuta el circuito construido por el usuario
- **Funcionalidad:**
  1. Valida que existe un circuito
  2. Clona el circuito para preservar el original
  3. Añade mediciones automáticas si es necesario
  4. Llama a `execute_circuit()` para la ejecución

##### `execute_circuit(qc, backend_name, shots)`
- **Propósito:** Lógica genérica de ejecución
- **Proceso:**
  1. Selecciona backend (especificado o menos ocupado)
  2. Crea instancia de `SamplerV2`
  3. Envía el trabajo
  4. Monitorea y presenta resultados con visualización de barras

## Ejemplos de Uso Completos

### Ejemplo 1: Circuito de Bell

```bash
Qiskit-CLI > crear 2
✅ Circuito de 2 qubits creado. ¡Listo para agregar puertas!

Qiskit-CLI > agregar h 0
✅ Puerta 'H' añadida a qubit(s) [0] con params [].

Qiskit-CLI > agregar cx 0 1
✅ Puerta 'CX' añadida a qubit(s) [0, 1] con params [].

Qiskit-CLI > ver
--- Circuito Actual ---
     ┌───┐     
q_0: ┤ H ├──■──
     └───┘┌─┴─┐
q_1: ─────┤ X ├
          └───┘

Qiskit-CLI > ejecutar_custom
🎯 Usando backend: ibm_brisbane
📊 Resultados de la medición:
  00: 512   (50.00%) ████████████████████████
  11: 512   (50.00%) ████████████████████████
```

### Ejemplo 2: Circuito con Rotaciones

```bash
Qiskit-CLI > crear 1
Qiskit-CLI > agregar rx pi/2 0
Qiskit-CLI > agregar ry pi/4 0
Qiskit-CLI > agregar rz pi/8 0
Qiskit-CLI > ver
--- Circuito Actual ---
     ┌─────────┐┌─────────┐┌─────────┐
q_0: ┤ Rx(π/2) ├┤ Ry(π/4) ├┤ Rz(π/8) ├
     └─────────┘└─────────┘└─────────┘
```

## Sistema de Logging

### Configuración
- **Archivo:** `~/.qiskit_cli/logs/qiskit_cli_YYYYMMDD.log`
- **Nivel por defecto:** INFO
- **Formato:** `TIMESTAMP | LEVEL | COMPONENT | MESSAGE`

### Personalización
```bash
python qiskit_cli.py --log-level DEBUG --log-file custom.log
```

## Comandos de Utilidad

### Navegación y Ayuda
```
ayuda          # Muestra ayuda completa
limpiar        # Limpia la pantalla
salir          # Sale de la aplicación
```

### Información del Sistema
```
probar         # Verifica conectividad
listar         # Lista backends disponibles
mejor          # Encuentra backend óptimo
```

## Manejo de Errores

La CLI incluye manejo robusto de errores:

- **Validación de entrada:** Verifica tipos y rangos
- **Errores de conexión:** Mensajes informativos sobre problemas de red
- **Errores de backend:** Gestión de backends no disponibles
- **Sintaxis de circuitos:** Validación de puertas y parámetros

## Contribución y Desarrollo

### Estructura del Código
- **Interfaz:** Función `interactive_shell()`
- **Lógica de negocio:** Clase `QiskitCLI`
- **Utilidades:** Clases `Colors` y funciones de logging

### Extensibilidad
El sistema está diseñado para fácil extensión:
- Nuevas puertas en `supported_gates` dictionary
- Comandos adicionales en `interactive_shell()`
- Backends personalizados a través de la API de Qiskit

---

**Versión:** 2.1.0 - Constructor de Circuitos Interactivo  
**Desarrollado con:** Python 3.x, Qiskit, Qiskit IBM Runtime  
**Licencia:** Consultar archivos del proyecto
CLI interactiva conectada a hardware cuantico real de ibm
