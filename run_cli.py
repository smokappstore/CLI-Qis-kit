#!/usr/bin/env python3
"""
Qiskit Runtime CLI - Herramienta de línea de comandos para gestionar servicios de Qiskit Runtime
by SmokAppSoftware jako with Claude AI, Gemini AI, COPILOT
Versión 2.2 - Constructor de Circuitos Interactivo con Simulador Local
"""

import argparse
import sys
import json
import os
import time
from datetime import datetime
from typing import Optional, List
import math # Importamos math para usar pi
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    print("⚠️  IBM Runtime no disponible. Funcionando solo con simulador local.")
    
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# --- Clase para colores en la consola ---
class Colors:
    """Clase para definir colores de texto ANSI para la consola."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configura el sistema de logging"""
    log_dir = Path.home() / ".qiskit_cli" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logger = logging.getLogger('qiskit_cli')
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter(
        f"{Colors.FAIL}%(levelname)-8s | %(message)s{Colors.ENDC}"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if not log_file:
        log_file = log_dir / f"qiskit_cli_{datetime.now().strftime('%Y%m%d')}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def print_banner():
    """Imprime el banner de la aplicación"""
    banner = f"""
{Colors.OKCYAN}{chr(9556)}{chr(9552)*76}{chr(9559)}
{chr(9553)}{'':^76}{chr(9553)}
{chr(9553)}{' ___    _         _      _   _       ____ _     ___ ':^76}{chr(9553)}
{chr(9553)}{' / _ \\  (_)  ___  | | __ (_) | |_    / ___| |   |_ _|':^76}{chr(9553)}
{chr(9553)}{'| | | | | | / __| | |/ / | | | __|  | |   | |    | | ':^76}{chr(9553)}
{chr(9553)}{'| |_| | | | \\__ \\ |   <  | | | |_   | |___| |___ | | ':^76}{chr(9553)}
{chr(9553)}{' \\__\\_\\ |_| |___/ |_|\\_\\ |_|  \\__|   \\____|_____|___|':^76}{chr(9553)}
{chr(9553)}{'':^76}{chr(9553)}
{chr(9553)}{'🚀 Qiskit Runtime CLI - Herramienta Cuántica Interactiva':^76}{chr(9553)}
{chr(9553)}{'by SmokAppSoftware jako with Claude AI & Gemini AI':^76}{chr(9553)}
{chr(9553)}{'🎯 Ahora con Simulador Local - ¡Sin credenciales!':^76}{chr(9553)}
{chr(9553)}{'':^76}{chr(9553)}
{chr(9562)}{chr(9552)*76}{chr(9565)}{Colors.ENDC}

{Colors.BOLD}⚡ Versión: 2.2.0 (Constructor + Simulador Local)
📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{Colors.OKGREEN}🌟 Escribe 'ayuda' para ver los comandos o 'salir' para terminar.
🎓 Prueba 'demo' para ver ejemplos de circuitos cuánticos!{Colors.ENDC}
"""
    print(banner)


class QiskitCLI:
    def __init__(self, logger: logging.Logger):
        self.service: Optional = None
        self.logger = logger
        # --- Atributo para guardar el circuito que estamos construyendo ---
        self.current_circuit: Optional[QuantumCircuit] = None
        # --- NUEVO: Simulador local ---
        self.local_simulator = AerSimulator()
        self.logger.info("Inicializando Qiskit CLI con simulador local")

    def _initialize_service(self):
        """Inicializa el servicio si aún no está activo."""
        if not IBM_AVAILABLE:
            print(f"{Colors.WARNING}⚠️ Funcionando solo con simulador local. Para usar IBM Quantum, instala qiskit-ibm-runtime.{Colors.ENDC}")
            return
            
        if not self.service:
            try:
                self.logger.debug("Inicializando servicio con credenciales guardadas.")
                self.service = QiskitRuntimeService(channel="ibm_quantum")
            except Exception as e:
                print(f"{Colors.FAIL}❌ No se pudo conectar a IBM Quantum. Usando simulador local.{Colors.ENDC}")
                self.logger.error(f"Fallo al inicializar el servicio: {e}", exc_info=True)
                self.service = None

    # --- IBM Quantum functions (mantienen la funcionalidad original) ---
    def setup_account(self, token: str, channel: str = "ibm_quantum") -> bool:
        """Configura y guarda la cuenta de Qiskit Runtime"""
        if not IBM_AVAILABLE:
            print(f"{Colors.WARNING}IBM Runtime no está disponible. Instala: pip install qiskit-ibm-runtime{Colors.ENDC}")
            return False
            
        try:
            self.logger.info(f"Configurando cuenta con canal: {channel}")
            QiskitRuntimeService.save_account(token=token, channel=channel, overwrite=True)
            success_msg = f"{Colors.OKGREEN}✅ ¡Cuenta configurada exitosamente!{Colors.ENDC}"
            print(success_msg)
            print("📝 Recuerda ir a tu dashboard (https://quantum.ibm.com/) y expirar tu API key anterior si es necesario.")
            self.logger.info("Cuenta configurada exitosamente.")
            self.service = None 
            return True
        except Exception as e:
            error_msg = f"❌ Error al configurar la cuenta: {e}"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            self.logger.error(f"Error al configurar cuenta: {e}", exc_info=True)
            return False

    def test_connection(self) -> bool:
        """Prueba la conexión con el servicio"""
        if not IBM_AVAILABLE:
            print(f"{Colors.OKBLUE}🖥️ Probando simulador local...{Colors.ENDC}")
            try:
                # Crear un circuito simple para probar el simulador
                qc = QuantumCircuit(1, 1)
                qc.h(0)
                qc.measure(0, 0)
                job = self.local_simulator.run(qc, shots=10)
                result = job.result()
                print(f"{Colors.OKGREEN}✅ ¡Simulador local funcionando correctamente!{Colors.ENDC}")
                return True
            except Exception as e:
                print(f"{Colors.FAIL}❌ Error con el simulador local: {e}{Colors.ENDC}")
                return False
        
        try:
            print("📡 Probando conexión con IBM Quantum...")
            self.logger.info("Iniciando prueba de conexión")
            start_time = time.time()
            self._initialize_service() 
            if not self.service:
                return False
            backends = self.service.backends()
            connection_time = time.time() - start_time
            success_msg = f"✅ ¡Conexión exitosa en {connection_time:.2f}s! Encontrados {len(backends)} backends."
            print(f"{Colors.OKGREEN}{success_msg}{Colors.ENDC}")
            self.logger.info(f"Conexión exitosa en {connection_time:.2f}s")
            return True
        except Exception as e:
            error_msg = f"❌ Error de conexión: {e}"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            self.logger.error(f"Error de conexión: {e}", exc_info=True)
            return False

    def list_backends(self, operational: bool = True, simulator: Optional[bool] = None):
        """Lista los backends disponibles"""
        print(f"\n{Colors.BOLD}📊 Backends disponibles:{Colors.ENDC}")
        print("-" * 78)
        
        # Siempre mostrar el simulador local
        print(f"{'#':<3} {'Nombre':<25} {'Tipo':<15} {'Estado':<18} {'Info Extra'}")
        print("-" * 78)
        print(f"{'1':<3} {'local_simulator':<25} {'🖥️  Simulador':<15} {Colors.OKGREEN + '🟢 Disponible' + Colors.ENDC:<28} {'Ilimitado qubits'}")
        
        if not IBM_AVAILABLE:
            print("-" * 78)
            print(f"{Colors.WARNING}Para acceder a backends de IBM Quantum, instala: pip install qiskit-ibm-runtime{Colors.ENDC}")
            return
        
        try:
            self.logger.info(f"Listando backends - Operacional: {operational}, Simulador: {simulator}")
            self._initialize_service()
            if not self.service:
                print("-" * 78)
                return
                
            backends = self.service.backends(operational=operational, simulator=simulator)

            if not backends:
                print(f"{Colors.WARNING}No se encontraron backends IBM con los filtros aplicados.{Colors.ENDC}")
                print("-" * 78)
                return

            for i, backend in enumerate(backends, 2):  # Empezar desde 2 porque el local es 1
                status = f"{Colors.OKGREEN}🟢 Operacional{Colors.ENDC}" if backend.operational else f"{Colors.FAIL}🔴 No operacional{Colors.ENDC}"
                backend_type = "🖥️  Simulador" if backend.simulator else "⚛️  Cuántico"
                extra_info = f"{backend.num_qubits} qubits, Cola: {backend.status().pending_jobs}"

                print(f"{i:<3} {backend.name:<25} {backend_type:<15} {status:<28} {extra_info}")
            print("-" * 78)
        except Exception as e:
            error_msg = f"❌ Error al listar backends IBM: {e}"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            self.logger.error(f"Error al listar backends: {e}", exc_info=True)
            print("-" * 78)

    def get_least_busy_backend(self, simulator: bool = False):
        """Obtiene el backend menos ocupado"""
        if not IBM_AVAILABLE or not self.service:
            print(f"{Colors.OKBLUE}🎯 Usando simulador local (siempre disponible){Colors.ENDC}")
            return "local_simulator"
        
        try:
            self.logger.info(f"Buscando backend menos ocupado - Simulador: {simulator}")
            self._initialize_service()
            if not self.service:
                print(f"{Colors.OKBLUE}🎯 Usando simulador local como fallback{Colors.ENDC}")
                return "local_simulator"

            print(f"🔎 Buscando el {'simulador' if simulator else 'computador cuántico'} menos ocupado...")
            backend = self.service.least_busy(operational=True, simulator=simulator)
            result_msg = f"🎯 Backend menos ocupado: {Colors.OKCYAN}{backend.name}{Colors.ENDC} (Cola: {backend.status().pending_jobs} trabajos)"
            print(result_msg)
            self.logger.info(f"Backend menos ocupado encontrado: {backend.name}")
            return backend
        except Exception as e:
            error_msg = f"❌ Error al obtener backend: {e}. Usando simulador local."
            print(f"{Colors.WARNING}{error_msg}{Colors.ENDC}")
            self.logger.error(f"Error al obtener backend: {e}", exc_info=True)
            return "local_simulator"

    # --- NUEVAS FUNCIONES DE DEMOSTRACIÓN ---
    
    def run_demo_circuits(self):
        """Ejecuta una serie de circuitos de demostración educativos"""
        demos = [
            ("🔄 Estado de Superposición (Hadamard)", self._demo_superposition),
            ("🔗 Entrelazamiento Cuántico (Bell)", self._demo_bell_state),
            ("🎯 Puertas de Pauli (X, Y, Z)", self._demo_pauli_gates),
            ("🌀 Puertas de Fase (S, T)", self._demo_phase_gates),
            ("🔄 Rotaciones Cuánticas", self._demo_rotation_gates),
            ("⚡ Circuito Cuántico Complejo", self._demo_complex_circuit)
        ]
        
        print(f"\n{Colors.BOLD}🎓 === DEMOSTRACIONES DE CIRCUITOS CUÁNTICOS === {Colors.ENDC}")
        print("¡Aprende cómo funcionan las principales puertas cuánticas!")
        print("-" * 60)
        
        for i, (name, demo_func) in enumerate(demos, 1):
            print(f"\n{Colors.OKCYAN}[{i}/6] {name}{Colors.ENDC}")
            print("─" * 50)
            try:
                demo_func()
                input(f"\n{Colors.OKGREEN}Presiona Enter para continuar con la siguiente demo...{Colors.ENDC}")
            except KeyboardInterrupt:
                print(f"\n{Colors.OKBLUE}Demo cancelada por el usuario.{Colors.ENDC}")
                return
            except Exception as e:
                print(f"{Colors.FAIL}Error en la demo: {e}{Colors.ENDC}")
                continue
        
        print(f"\n{Colors.BOLD}{Colors.OKGREEN}🎉 ¡Todas las demos completadas! Ahora puedes crear tus propios circuitos.{Colors.ENDC}")
        print(f"💡 Usa 'crear <qubits>' para empezar tu propio circuito.")

    def _demo_superposition(self):
        """Demuestra el estado de superposición con puerta Hadamard"""
        print("📚 La puerta Hadamard crea una superposición: |0⟩ y |1⟩ con igual probabilidad")
        
        qc = QuantumCircuit(1, 1)
        qc.h(0)  # Puerta Hadamard
        qc.measure(0, 0)
        
        print(f"\n{Colors.OKBLUE}Circuito:{Colors.ENDC}")
        print(qc.draw(output='text'))
        
        print(f"\n{Colors.OKBLUE}🔬 Ejecutando 1000 mediciones...{Colors.ENDC}")
        job = self.local_simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        
        print(f"{Colors.BOLD}📊 Resultados:{Colors.ENDC}")
        self._display_results(counts, 1000)
        print("💡 Observa que obtienes aproximadamente 50% de |0⟩ y 50% de |1⟩")

    def _demo_bell_state(self):
        """Demuestra el entrelazamiento cuántico con estado de Bell"""
        print("📚 El estado de Bell crea dos qubits entrelazados")
        print("   Los qubits siempre darán el mismo resultado al medirlos!")
        
        qc = QuantumCircuit(2, 2)
        qc.h(0)      # Superposición en el primer qubit
        qc.cx(0, 1)  # CNOT crea entrelazamiento
        qc.measure_all()
        
        print(f"\n{Colors.OKBLUE}Circuito:{Colors.ENDC}")
        print(qc.draw(output='text'))
        
        print(f"\n{Colors.OKBLUE}🔬 Ejecutando 1000 mediciones...{Colors.ENDC}")
        job = self.local_simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        
        print(f"{Colors.BOLD}📊 Resultados:{Colors.ENDC}")
        self._display_results(counts, 1000)
        print("💡 Solo obtienes |00⟩ y |11⟩ - ¡los qubits están entrelazados!")

    def _demo_pauli_gates(self):
        """Demuestra las puertas de Pauli X, Y, Z"""
        print("📚 Las puertas de Pauli son las operaciones cuánticas fundamentales")
        
        gates = [
            ("X (NOT cuántico)", "x", "Invierte |0⟩ ↔ |1⟩"),
            ("Y", "y", "Rotación compleja"),
            ("Z", "z", "Cambio de fase")
        ]
        
        for gate_name, gate_method, description in gates:
            print(f"\n🔸 Puerta {gate_name}: {description}")
            qc = QuantumCircuit(1, 1)
            getattr(qc, gate_method)(0)
            qc.measure(0, 0)
            
            print(f"{Colors.OKBLUE}Circuito:{Colors.ENDC}")
            print(qc.draw(output='text'))
            
            job = self.local_simulator.run(qc, shots=1000)
            result = job.result()
            counts = result.get_counts(qc)
            
            print(f"{Colors.BOLD}📊 Resultados:{Colors.ENDC}")
            self._display_results(counts, 1000, compact=True)

    def _demo_phase_gates(self):
        """Demuestra las puertas de fase S y T"""
        print("📚 Las puertas de fase S y T añaden fases cuánticas")
        print("   (No visibles directamente, pero importantes para algoritmos)")
        
        # Demostración con interferencia
        qc = QuantumCircuit(1, 1)
        qc.h(0)    # Superposición
        qc.s(0)    # Puerta S (fase π/2)
        qc.h(0)    # Hadamard otra vez para mostrar interferencia
        qc.measure(0, 0)
        
        print(f"\n{Colors.OKBLUE}Circuito H-S-H (muestra efectos de fase):{Colors.ENDC}")
        print(qc.draw(output='text'))
        
        job = self.local_simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        
        print(f"{Colors.BOLD}📊 Resultados:{Colors.ENDC}")
        self._display_results(counts, 1000)
        print("💡 La fase S cambió la interferencia - ¡más probabilidad de |1⟩!")

    def _demo_rotation_gates(self):
        """Demuestra las puertas de rotación RX, RY, RZ"""
        print("📚 Las puertas de rotación permiten control preciso del estado cuántico")
        
        angles = [("π/4", math.pi/4), ("π/2", math.pi/2), ("π", math.pi)]
        
        for angle_name, angle_val in angles:
            print(f"\n🔸 Rotación RY({angle_name}):")
            qc = QuantumCircuit(1, 1)
            qc.ry(angle_val, 0)
            qc.measure(0, 0)
            
            print(f"{Colors.OKBLUE}Circuito:{Colors.ENDC}")
            print(qc.draw(output='text'))
            
            job = self.local_simulator.run(qc, shots=1000)
            result = job.result()
            counts = result.get_counts(qc)
            
            print(f"{Colors.BOLD}📊 Resultados:{Colors.ENDC}")
            self._display_results(counts, 1000, compact=True)

    def _demo_complex_circuit(self):
        """Demuestra un circuito cuántico más complejo"""
        print("📚 Circuito complejo: Superposición + Entrelazamiento + Rotaciones")
        
        qc = QuantumCircuit(3, 3)
        # Crear superposición en todos los qubits
        qc.h([0, 1, 2])
        # Entrelazar qubits 0 y 1
        qc.cx(0, 1)
        # Aplicar rotación condicional
        qc.crz(math.pi/4, 1, 2)
        # Puerta Toffoli (CCNOT)
        qc.ccx(0, 1, 2)
        qc.measure_all()
        
        print(f"\n{Colors.OKBLUE}Circuito:{Colors.ENDC}")
        print(qc.draw(output='text'))
        
        print(f"\n{Colors.OKBLUE}🔬 Ejecutando 1000 mediciones...{Colors.ENDC}")
        job = self.local_simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        
        print(f"{Colors.BOLD}📊 Resultados:{Colors.ENDC}")
        self._display_results(counts, 1000)
        print("💡 ¡Circuito complejo con múltiples correlaciones cuánticas!")

    def _display_results(self, counts, shots, compact=False):
        """Muestra los resultados de manera visual"""
        sorted_counts = dict(sorted(counts.items()))
        
        for outcome, count in sorted_counts.items():
            percentage = (count / shots) * 100
            if compact:
                bar = '█' * max(1, int(percentage / 5))  # Barras más cortas para modo compacto
                print(f"  |{outcome}⟩: {count:>4} ({percentage:5.1f}%) {Colors.OKBLUE}{bar}{Colors.ENDC}")
            else:
                bar = '█' * max(1, int(percentage / 2))
                print(f"  |{outcome}⟩: {count:>4} ({percentage:5.1f}%) {Colors.OKBLUE}{bar}{Colors.ENDC}")

    # --- Constructor de Circuitos (funciones originales mejoradas) ---
    
    def create_circuit(self, num_qubits_str: str):
        """Inicia un nuevo circuito cuántico."""
        try:
            num_qubits = int(num_qubits_str)
            if num_qubits <= 0:
                print(f"{Colors.FAIL}El número de qubits debe ser mayor que cero.{Colors.ENDC}")
                return
            if num_qubits > 20:
                print(f"{Colors.WARNING}⚠️ {num_qubits} qubits es mucho para visualizar. Recomendado: ≤ 10 qubits.{Colors.ENDC}")
            
            self.current_circuit = QuantumCircuit(num_qubits)
            self.logger.info(f"Nuevo circuito creado con {num_qubits} qubits.")
            print(f"{Colors.OKGREEN}✅ Circuito de {num_qubits} qubits creado. ¡Listo para agregar puertas!{Colors.ENDC}")
            print(f"   Usa {Colors.OKCYAN}agregar <puerta> <qubits...>{Colors.ENDC} para continuar.")
            print(f"   Ejemplo: {Colors.OKCYAN}agregar h 0{Colors.ENDC} (Hadamard en qubit 0)")
        except ValueError:
            print(f"{Colors.FAIL}Entrada inválida. '{num_qubits_str}' no es un número entero.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Error al crear el circuito: {e}{Colors.ENDC}")

    def add_gate_to_circuit(self, args: List[str]):
        """Añade una puerta cuántica al circuito actual."""
        if not self.current_circuit:
            print(f"{Colors.WARNING}⚠️ No hay ningún circuito activo. Usa `crear <num_qubits>` primero.{Colors.ENDC}")
            return

        if not args:
            print(f"{Colors.FAIL}Debes especificar una puerta y sus qubits.{Colors.ENDC}")
            self._show_gate_help()
            return

        gate_name = args[0].lower()
        gate_args = args[1:]

        # Mapa extendido de puertas soportadas
        supported_gates = {
            # Puertas de 1 qubit
            'h': ('h', 1, 0, 'Hadamard - crea superposición'), 
            'x': ('x', 1, 0, 'Pauli-X - NOT cuántico'), 
            'y': ('y', 1, 0, 'Pauli-Y - rotación compleja'), 
            'z': ('z', 1, 0, 'Pauli-Z - cambio de fase'),
            's': ('s', 1, 0, 'S - fase π/2'), 
            'sdg': ('sdg', 1, 0, 'S† - fase -π/2'), 
            't': ('t', 1, 0, 'T - fase π/4'), 
            'tdg': ('tdg', 1, 0, 'T† - fase -π/4'),
            'i': ('i', 1, 0, 'Identidad - no hace nada'),
            # Puertas de 2 qubits
            'cx': ('cx', 2, 0, 'CNOT - entrelazamiento'), 
            'cz': ('cz', 2, 0, 'Controlled-Z'), 
            'swap': ('swap', 2, 0, 'SWAP - intercambia qubits'),
            'cy': ('cy', 2, 0, 'Controlled-Y'),
            # Puertas de rotación
            'rx': ('rx', 1, 1, 'Rotación-X - rx(θ, qubit)'), 
            'ry': ('ry', 1, 1, 'Rotación-Y - ry(θ, qubit)'), 
            'rz': ('rz', 1, 1, 'Rotación-Z - rz(θ, qubit)'), 
            'p': ('p', 1, 1, 'Fase - p(λ, qubit)'),
            'u': ('u', 1, 3, 'U universal - u(θ,φ,λ, qubit)'),
            # Puertas controladas
            'crx': ('crx', 2, 1, 'Controlled-RX - crx(θ, c, t)'),
            'cry': ('cry', 2, 1, 'Controlled-RY - cry(θ, c, t)'),
            'crz': ('crz', 2, 1, 'Controlled-RZ - crz(θ, c, t)'),
            'cp': ('cp', 2, 1, 'Controlled-Phase - cp(λ, c, t)'),
            # Puertas de 3 qubits
            'ccx': ('ccx', 3, 0, 'Toffoli - doble control'),
            'cswap': ('cswap', 3, 0, 'Fredkin - SWAP controlado')
        }

        if gate_name not in supported_gates:
            print(f"{Colors.FAIL}Puerta '{gate_name}' no reconocida.{Colors.ENDC}")
            self._show_gate_help()
            return

        method_name, num_qubits_req, num_params_req, description = supported_gates[gate_name]

        if len(gate_args) != num_qubits_req + num_params_req:
            print(f"{Colors.FAIL}Número incorrecto de argumentos para la puerta '{gate_name}'.{Colors.ENDC}")
            print(f"   Uso: {Colors.OKCYAN}agregar {gate_name} {description.split('-')[1].strip()}{Colors.ENDC}")
            return
        
        try:
            # --- LÓGICA COMPLETADA ---
            params = []
            # 1. Extraer y evaluar parámetros (ángulos)
            param_strings = gate_args[:num_params_req]
            for p_str in param_strings:
                # Permitir expresiones como 'pi/2'
                safe_dict = {"pi": math.pi, "np": np}
                try:
                    params.append(eval(p_str, {"__builtins__": {}}, safe_dict))
                except Exception:
                    print(f"{Colors.FAIL}Error al evaluar el parámetro '{p_str}'. Usa números o 'pi'.{Colors.ENDC}")
                    return

            # 2. Extraer y validar qubits
            qubit_strings = gate_args[num_params_req:]
            qubits = [int(q) for q in qubit_strings]
            
            for q in qubits:
                if not 0 <= q < self.current_circuit.num_qubits:
                    print(f"{Colors.FAIL}Qubit inválido: {q}. El circuito tiene {self.current_circuit.num_qubits} qubits (0 a {self.current_circuit.num_qubits-1}).{Colors.ENDC}")
                    return

            # 3. Obtener el método de la puerta y aplicarla
            gate_method = getattr(self.current_circuit, method_name)
            
            # 4. Llamar al método con los argumentos correctos
            # Las puertas de rotación esperan (ángulo, qubit), las de control (ángulo, control, target), etc.
            # Las puertas simples esperan (qubit)
            full_args = params + qubits
            gate_method(*full_args)

            print(f"{Colors.OKGREEN}✅ Puerta '{gate_name.upper()}' añadida.{Colors.ENDC}")
            self.show_circuit([]) # Mostrar el circuito actualizado

        except ValueError:
            print(f"{Colors.FAIL}Error: Los qubits deben ser números enteros.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Error inesperado al añadir la puerta: {e}{Colors.ENDC}")
            self.logger.error(f"Error en add_gate: {e}", exc_info=True)


    def _show_gate_help(self):
        """Muestra una lista de las puertas disponibles."""
        print(f"\n{Colors.BOLD}📖 Puertas Cuánticas Disponibles:{Colors.ENDC}")
        print("Uso: agregar <nombre_puerta> [parámetros...] <qubits...>")
        print("-" * 60)
        # Aquí puedes añadir más puertas si quieres
        gate_info = {
            "1 Qubit": ["h", "x", "y", "z", "s", "t", "rx(theta, q)", "ry(theta, q)"],
            "2 Qubits": ["cx(c, t)", "cz(c, t)", "swap(q1, q2)", "crz(theta, c, t)"],
            "3 Qubits": ["ccx(c1, c2, t)", "cswap(c, t1, t2)"]
        }
        for category, gates in gate_info.items():
            print(f"{Colors.OKCYAN}{category}:{Colors.ENDC}")
            for gate in gates:
                print(f"  - {gate}")
        print("-" * 60)
        print("Ejemplo: agregar rx pi/2 0  (Rotación de 90° en qubit 0)")
        print("Ejemplo: agregar cx 0 1     (CNOT con control en 0 y target en 1)")


    def show_circuit(self, args: List[str]):
        """Muestra el circuito cuántico actual."""
        if not self.current_circuit:
            print(f"{Colors.WARNING}⚠️ No hay ningún circuito activo. Usa `crear <num_qubits>` primero.{Colors.ENDC}")
            return
        
        print(f"\n{Colors.BOLD}--- 📝 Circuito Actual ---{Colors.ENDC}")
        print(f"Qubits: {self.current_circuit.num_qubits}, Bits Clásicos: {self.current_circuit.num_clbits}, Profundidad: {self.current_circuit.depth()}")
        print(self.current_circuit.draw(output='text'))
        print(f"{Colors.BOLD}--------------------------{Colors.ENDC}")


    def add_measurement(self, args: List[str]):
        """Añade mediciones al circuito."""
        if not self.current_circuit:
            print(f"{Colors.WARNING}⚠️ No hay ningún circuito activo.{Colors.ENDC}")
            return
        
        if not args or args[0].lower() == 'all':
            # Medir todos los qubits en bits clásicos correspondientes
            if self.current_circuit.num_clbits < self.current_circuit.num_qubits:
                self.current_circuit.add_bits(self.current_circuit.num_qubits - self.current_circuit.num_clbits)
            self.current_circuit.measure_all()
            print(f"{Colors.OKGREEN}✅ Medición añadida para todos los {self.current_circuit.num_qubits} qubits.{Colors.ENDC}")
        else:
            try:
                qubits_to_measure = [int(q) for q in args]
                max_q = max(qubits_to_measure)
                if max_q >= self.current_circuit.num_qubits:
                    print(f"{Colors.FAIL}Error: Qubit {max_q} fuera de rango.{Colors.ENDC}")
                    return
                # Añadir bits clásicos si es necesario
                if self.current_circuit.num_clbits < len(qubits_to_measure):
                     self.current_circuit.add_bits(len(qubits_to_measure) - self.current_circuit.num_clbits)
                
                self.current_circuit.measure(qubits_to_measure, range(len(qubits_to_measure)))
                print(f"{Colors.OKGREEN}✅ Medición añadida para qubits {qubits_to_measure}.{Colors.ENDC}")

            except ValueError:
                print(f"{Colors.FAIL}Error: Los qubits a medir deben ser números enteros o 'all'.{Colors.ENDC}")
        
        self.show_circuit([])


    def run_circuit(self, args: List[str]):
        """Ejecuta el circuito actual."""
        if not self.current_circuit:
            print(f"{Colors.WARNING}⚠️ No hay ningún circuito para ejecutar.{Colors.ENDC}")
            return

        if self.current_circuit.num_clbits == 0:
            print(f"{Colors.WARNING}⚠️ El circuito no tiene mediciones. Añadiendo 'measure_all()' para obtener resultados.{Colors.ENDC}")
            self.add_measurement(['all'])

        # Parsear argumentos: [backend] [shots]
        backend_name = "local_simulator"
        shots = 1024
        if len(args) > 0:
            backend_name = args[0]
        if len(args) > 1:
            try:
                shots = int(args[1])
            except ValueError:
                print(f"{Colors.FAIL}Error: El número de shots debe ser un entero.{Colors.ENDC}")
                return

        print(f"\n{Colors.BOLD}🚀 Ejecutando circuito...{Colors.ENDC}")
        print(f"   Backend: {Colors.OKCYAN}{backend_name}{Colors.ENDC}")
        print(f"   Shots: {Colors.OKCYAN}{shots}{Colors.ENDC}")
        
        try:
            if backend_name == "local_simulator":
                transpiled_circuit = transpile(self.current_circuit, self.local_simulator)
                job = self.local_simulator.run(transpiled_circuit, shots=shots)
                result = job.result()
                counts = result.get_counts(transpiled_circuit)
            else:
                if not IBM_AVAILABLE:
                    print(f"{Colors.FAIL}IBM Runtime no disponible. No se puede ejecutar en '{backend_name}'.{Colors.ENDC}")
                    return
                self._initialize_service()
                if not self.service:
                    return
                backend = self.service.get_backend(backend_name)
                sampler = Sampler(backend)
                job = sampler.run(self.current_circuit, shots=shots)
                print(f"   Job ID (IBM): {Colors.OKBLUE}{job.job_id()}{Colors.ENDC}")
                print("   Esperando resultados de IBM Quantum...")
                result = job.result()
                # El resultado de SamplerV2 está en pub_results
                pub_result = result[0]
                # Los datos binarios se convierten a counts
                counts = pub_result.data.c.get_counts()

            print(f"\n{Colors.BOLD}📊 Resultados de la ejecución:{Colors.ENDC}")
            self._display_results(counts, shots)
            
            # Generar y mostrar histograma
            plot_histogram(counts)
            save_path = Path.home() / ".qiskit_cli" / f"histogram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path)
            print(f"\n{Colors.OKGREEN}📈 Histograma guardado en: {save_path}{Colors.ENDC}")
            plt.show() # Muestra el gráfico en una ventana

        except Exception as e:
            print(f"{Colors.FAIL}❌ Error durante la ejecución: {e}{Colors.ENDC}")
            self.logger.error(f"Error en run_circuit: {e}", exc_info=True)


    def clear_circuit(self, args: List[str]):
        """Limpia el circuito actual."""
        if self.current_circuit:
            self.current_circuit = None
            print(f"{Colors.OKBLUE}🗑️ Circuito actual eliminado.{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}No hay ningún circuito activo para eliminar.{Colors.ENDC}")

    def _show_interactive_help(self):
        """Muestra la ayuda para los comandos interactivos."""
        print(f"\n{Colors.BOLD}--- 💡 Ayuda de Comandos Interactivos ---{Colors.ENDC}")
        commands = {
            "Construcción de Circuitos": {
                "crear <num_qubits>": "Inicia un nuevo circuito.",
                "agregar <puerta> ...": "Añade una puerta. Escribe 'agregar' para ver la lista de puertas.",
                "medir [all|q1 q2..]": "Añade mediciones al circuito.",
                "ver": "Muestra el circuito actual en formato de texto.",
                "limpiar": "Elimina el circuito actual."
            },
            "Ejecución": {
                "ejecutar [backend] [shots]": "Ejecuta el circuito (def: local_simulator, 1024 shots)."
            },
            "IBM Quantum": {
                "backends": "Lista los backends de IBM y el simulador local.",
                "test": "Prueba la conexión a IBM Quantum y el simulador local."
            },
            "Otros": {
                "demo": "Ejecuta una serie de circuitos de demostración.",
                "ayuda": "Muestra esta ayuda.",
                "salir": "Sale de la aplicación."
            }
        }
        for category, cmds in commands.items():
            print(f"\n{Colors.OKCYAN}{category}:{Colors.ENDC}")
            for cmd, desc in cmds.items():
                print(f"  {Colors.BOLD}{cmd:<25}{Colors.ENDC} {desc}")
        print(f"{Colors.BOLD}------------------------------------------{Colors.ENDC}")
    
    def run_interactive_mode(self):
        """Inicia el bucle de la línea de comandos interactiva."""
        
        command_map = {
            'crear': self.create_circuit,
            'agregar': self.add_gate_to_circuit,
            'ver': self.show_circuit,
            'medir': self.add_measurement,
            'ejecutar': self.run_circuit,
            'limpiar': self.clear_circuit,
            'backends': lambda args: self.list_backends(),
            'test': lambda args: self.test_connection(),
            'demo': lambda args: self.run_demo_circuits(),
            'ayuda': lambda args: self._show_interactive_help(),
            'salir': lambda args: sys.exit(0)
        }

        while True:
            try:
                prompt_status = f"{self.current_circuit.num_qubits}Q" if self.current_circuit else "vacío"
                prompt = f"{Colors.BOLD}{Colors.OKGREEN}(circuito: {prompt_status}) »{Colors.ENDC} "
                user_input = input(prompt).strip()
                if not user_input:
                    continue

                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:]

                if command in command_map:
                    if command == 'crear':
                        if len(args) == 1:
                            command_map[command](args[0])
                        else:
                            print(f"{Colors.FAIL}Uso: crear <num_qubits>{Colors.ENDC}")
                    else:
                        command_map[command](args)
                else:
                    print(f"{Colors.FAIL}Comando '{command}' desconocido. Escribe 'ayuda' para ver la lista.{Colors.ENDC}")

            except (KeyboardInterrupt, EOFError):
                print("\n👋 ¡Hasta luego!")
                sys.exit(0)
            except Exception as e:
                print(f"{Colors.FAIL}Ha ocurrido un error inesperado: {e}{Colors.ENDC}")
                self.logger.error(f"Error en el bucle interactivo: {e}", exc_info=True)


def main():
    """Función principal que maneja argumentos de línea de comandos y modo interactivo."""
    parser = argparse.ArgumentParser(description="Qiskit Runtime CLI - Herramienta para gestionar Qiskit Runtime.")
    parser.add_argument("--log-level", default="INFO", help="Nivel de logging (DEBUG, INFO, WARNING, ERROR)")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # Comando 'setup'
    setup_parser = subparsers.add_parser("setup", help="Configura tus credenciales de IBM Quantum.")
    setup_parser.add_argument("token", help="Tu token de API de IBM Quantum.")
    setup_parser.add_argument("--channel", default="ibm_quantum", help="Canal a usar (ej: ibm_quantum).")

    # Comando 'test'
    subparsers.add_parser("test", help="Prueba la conexión con los servicios de IBM Quantum.")

    # Comando 'list-backends'
    list_parser = subparsers.add_parser("list", help="Lista los backends disponibles.")
    list_parser.add_argument("--all", action="store_true", help="Muestra todos los backends, no solo los operacionales.")
    list_parser.add_argument("--type", choices=["sim", "real"], help="Filtra por tipo (simulador o real).")

    # Comando 'least-busy'
    busy_parser = subparsers.add_parser("least-busy", help="Encuentra el backend menos ocupado.")
    busy_parser.add_argument("--sim", action="store_true", help="Busca el simulador menos ocupado.")

    args = parser.parse_args()
    logger = setup_logging(log_level=args.log_level)
    cli = QiskitCLI(logger)

    if args.command:
        # Modo no interactivo
        if args.command == "setup":
            cli.setup_account(args.token, args.channel)
        elif args.command == "test":
            cli.test_connection()
        elif args.command == "list":
            sim_filter = None
            if args.type == "sim":
                sim_filter = True
            elif args.type == "real":
                sim_filter = False
            cli.list_backends(operational=not args.all, simulator=sim_filter)
        elif args.command == "least-busy":
            cli.get_least_busy_backend(simulator=args.sim)
    else:
        # Modo interactivo
        print_banner()
        cli.run_interactive_mode()

if __name__ == "__main__":
    main()