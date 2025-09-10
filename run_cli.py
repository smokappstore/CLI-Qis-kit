#!/usr/bin/env python3
"""
Qiskit Runtime CLI - Herramienta de línea de comandos para gestionar servicios de Qiskit Runtime
by SmokAppSoftware jako with Claude Ai, Gemini AI, COPILOT
Versión 2.1 - Constructor de Circuitos Interactivo
"""

import argparse
import sys
import json
import os
import time
from datetime import datetime
from typing import Optional, List
import math # Importamos math para usar pi
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import logging
from pathlib import Path

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
{chr(9553)}{'':^76}{chr(9553)}
{chr(9562)}{chr(9552)*76}{chr(9565)}{Colors.ENDC}

{Colors.BOLD}⚡ Versión: 2.1.0 (Constructor de Circuitos)
📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{Colors.OKGREEN}🌟 Escribe 'ayuda' para ver los comandos o 'salir' para terminar.{Colors.ENDC}
"""
    print(banner)


class QiskitCLI:
    def __init__(self, logger: logging.Logger):
        self.service: Optional[QiskitRuntimeService] = None
        self.logger = logger
        # --- NUEVO: Atributo para guardar el circuito que estamos construyendo ---
        self.current_circuit: Optional[QuantumCircuit] = None
        self.logger.info("Inicializando Qiskit CLI")
    
    def _initialize_service(self):
        """Inicializa el servicio si aún no está activo."""
        if not self.service:
            try:
                self.logger.debug("Inicializando servicio con credenciales guardadas.")
                self.service = QiskitRuntimeService(channel="ibm_quantum")
            except Exception as e:
                print(f"{Colors.FAIL}❌ No se pudo conectar. ¿Has configurado tu cuenta? "
                      f"Usa el comando 'configurar'.{Colors.ENDC}")
                self.logger.error(f"Fallo al inicializar el servicio: {e}", exc_info=True)
                raise

    # --- El resto de funciones (setup_account, test_connection, etc.) permanecen igual ---
    def setup_account(self, token: str, channel: str = "ibm_quantum") -> bool:
        """Configura y guarda la cuenta de Qiskit Runtime"""
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
        try:
            print("📡 Probando conexión con IBM Quantum...")
            self.logger.info("Iniciando prueba de conexión")
            start_time = time.time()
            self._initialize_service() 
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
        try:
            self.logger.info(f"Listando backends - Operacional: {operational}, Simulador: {simulator}")
            self._initialize_service()
            backends = self.service.backends(operational=operational, simulator=simulator)
            
            if not backends:
                print(f"{Colors.WARNING}No se encontraron backends con los filtros aplicados.{Colors.ENDC}")
                return

            print(f"\n{Colors.BOLD}📊 Backends disponibles:{Colors.ENDC}")
            print("-" * 78)
            print(f"{'#':<3} {'Nombre':<25} {'Tipo':<15} {'Estado':<18} {'Info Extra'}")
            print("-" * 78)
            
            for i, backend in enumerate(backends, 1):
                status = f"{Colors.OKGREEN}🟢 Operacional{Colors.ENDC}" if backend.operational else f"{Colors.FAIL}🔴 No operacional{Colors.ENDC}"
                backend_type = "🖥️  Simulador" if backend.simulator else "⚛️  Cuántico"
                extra_info = f"{backend.num_qubits} qubits, Cola: {backend.status().pending_jobs}"
                
                print(f"{i:<3} {backend.name:<25} {backend_type:<15} {status:<28} {extra_info}")
            print("-" * 78)
        except Exception as e:
            error_msg = f"❌ Error al listar backends: {e}"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            self.logger.error(f"Error al listar backends: {e}", exc_info=True)
    
    def get_least_busy_backend(self, simulator: bool = False):
        """Obtiene el backend menos ocupado"""
        try:
            self.logger.info(f"Buscando backend menos ocupado - Simulador: {simulator}")
            self._initialize_service()
            
            print(f"🔎 Buscando el {'simulador' if simulator else 'computador cuántico'} menos ocupado...")
            backend = self.service.least_busy(operational=True, simulator=simulator)
            result_msg = f"🎯 Backend menos ocupado: {Colors.OKCYAN}{backend.name}{Colors.ENDC} (Cola: {backend.status().pending_jobs} trabajos)"
            print(result_msg)
            self.logger.info(f"Backend menos ocupado encontrado: {backend.name}")
            return backend
        except Exception as e:
            error_msg = f"❌ Error al obtener backend: {e}"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            self.logger.error(f"Error al obtener backend: {e}", exc_info=True)
            return None

    # --- La función de ejecutar el ejemplo sigue igual por si se quiere usar ---
    def run_example_circuit(self, backend_name: Optional[str] = None, shots: int = 1024):
        """Ejecuta un circuito de Bell de ejemplo"""
        try:
            self.logger.info(f"Iniciando ejecución de circuito de ejemplo")
            # Crear circuito de Bell
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            
            print(f"\n{Colors.BOLD}🔧 Circuito de Bell a ejecutar:{Colors.ENDC}")
            # Llamamos a la función genérica de ejecución
            self.execute_circuit(qc, backend_name, shots)
            
        except Exception as e:
            error_msg = f"❌ Error al ejecutar circuito de ejemplo: {e}"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            self.logger.error(f"Error al ejecutar circuito de ejemplo: {e}", exc_info=True)
            return None
    
    def check_job_status(self, job_id: str):
        """Verifica el estado de un job"""
        try:
            self.logger.info(f"Verificando estado del job: {job_id}")
            self._initialize_service()
            
            print(f"🔎 Verificando estado del job {Colors.OKCYAN}{job_id}{Colors.ENDC}...")
            job = self.service.job(job_id)
            status = job.status()
            
            print(f"\n{Colors.BOLD}--- Estado del Job ---{Colors.ENDC}")
            print(f"📋 ID: {job_id}")
            print(f"📊 Estado: {Colors.OKCYAN}{status.name} - {status.value}{Colors.ENDC}")
            print(f"🖥️  Backend: {job.backend().name}")
            
            if hasattr(job, 'queue_position') and job.queue_position() is not None:
                print(f"🔢 Posición en cola: {job.queue_position()}")
                
            if hasattr(job, 'creation_date'):
                print(f"📅 Fecha de creación: {job.creation_date().strftime('%Y-%m-%d %H:%M:%S %Z')}")

            return status
        except Exception as e:
            error_msg = f"❌ Error al verificar job: {e}"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            self.logger.error(f"Error al verificar job {job_id}: {e}", exc_info=True)
            return None

    # --- INICIO DE NUEVAS FUNCIONES PARA EL CONSTRUCTOR DE CIRCUITOS ---

    def create_circuit(self, num_qubits_str: str):
        """Inicia un nuevo circuito cuántico."""
        try:
            num_qubits = int(num_qubits_str)
            if num_qubits <= 0:
                print(f"{Colors.FAIL}El número de qubits debe ser mayor que cero.{Colors.ENDC}")
                return
            self.current_circuit = QuantumCircuit(num_qubits)
            self.logger.info(f"Nuevo circuito creado con {num_qubits} qubits.")
            print(f"{Colors.OKGREEN}✅ Circuito de {num_qubits} qubits creado. ¡Listo para agregar puertas!{Colors.ENDC}")
            print(f"   Usa {Colors.OKCYAN}agregar <puerta> <qubits...>{Colors.ENDC} para continuar.")
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
            print(f"{Colors.FAIL}Debes especificar una puerta y sus qubits. Ej: `agregar h 0`{Colors.ENDC}")
            return
        
        gate_name = args[0].lower()
        gate_args = args[1:]

        # Mapa de puertas soportadas: {nombre: (método_qiskit, num_qubits, num_params)}
        supported_gates = {
            'h': ('h', 1, 0), 'x': ('x', 1, 0), 'y': ('y', 1, 0), 'z': ('z', 1, 0),
            's': ('s', 1, 0), 'sdg': ('sdg', 1, 0), 't': ('t', 1, 0), 'tdg': ('tdg', 1, 0),
            'cx': ('cx', 2, 0), 'cz': ('cz', 2, 0), 'swap': ('swap', 2, 0),
            'rx': ('rx', 1, 1), 'ry': ('ry', 1, 1), 'rz': ('rz', 1, 1), 'p': ('p', 1, 1)
        }

        if gate_name not in supported_gates:
            print(f"{Colors.FAIL}Puerta '{gate_name}' no reconocida. Puertas soportadas: {', '.join(supported_gates.keys())}{Colors.ENDC}")
            return

        method_name, num_qubits_req, num_params_req = supported_gates[gate_name]
        
        if len(gate_args) != num_qubits_req + num_params_req:
            print(f"{Colors.FAIL}Argumentos incorrectos para la puerta '{gate_name}'.")
            print(f"   Requiere {num_qubits_req} qubit(s) y {num_params_req} parámetro(s).{Colors.ENDC}")
            return

        try:
            # Separar qubits y parámetros
            qubit_indices_str = gate_args[:num_qubits_req]
            param_str = gate_args[num_qubits_req:]
            
            qubits = [int(q) for q in qubit_indices_str]
            # Evaluar parámetros: permite usar 'pi', 'pi/2', etc.
            params = [eval(p, {"__builtins__": None}, {"pi": math.pi}) for p in param_str]

            # Validar índices de qubits
            for q in qubits:
                if not (0 <= q < self.current_circuit.num_qubits):
                    print(f"{Colors.FAIL}Índice de qubit fuera de rango: {q}. El circuito tiene {self.current_circuit.num_qubits} qubits (0 a {self.current_circuit.num_qubits-1}).{Colors.ENDC}")
                    return

            # Obtener el método de la puerta del objeto circuito y llamarlo
            gate_method = getattr(self.current_circuit, method_name)
            gate_method(*params, *qubits) # Desempaquetar parámetros primero, luego qubits
            
            print(f"{Colors.OKGREEN}✅ Puerta '{gate_name.upper()}' añadida a qubit(s) {qubits} con params {params}.{Colors.ENDC}")
            self.logger.info(f"Puerta {gate_name} añadida al circuito: qubits={qubits}, params={params}")

        except ValueError:
            print(f"{Colors.FAIL}Error: Los qubits deben ser enteros y los parámetros deben ser números.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Error al agregar la puerta: {e}{Colors.ENDC}")

    def view_circuit(self):
        """Muestra el circuito actual en la consola."""
        if self.current_circuit and self.current_circuit.size() > 0:
            print(f"\n{Colors.BOLD}--- Circuito Actual ---{Colors.ENDC}")
            print(self.current_circuit.draw(output='text'))
        elif self.current_circuit:
            print(f"{Colors.OKBLUE}El circuito está vacío. Usa `agregar` para añadir puertas.{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}⚠️ No hay ningún circuito activo. Usa `crear <num_qubits>` para empezar.{Colors.ENDC}")

    def clear_circuit(self):
        """Limpia el circuito actual."""
        if self.current_circuit:
            self.current_circuit = None
            print(f"{Colors.OKBLUE}🗑️ Circuito actual eliminado.{Colors.ENDC}")
            self.logger.info("Circuito actual ha sido limpiado.")
        else:
            print(f"{Colors.WARNING}No hay ningún circuito para limpiar.{Colors.ENDC}")

    def run_custom_circuit(self, backend_name: Optional[str] = None, shots: int = 1024):
        """Ejecuta el circuito creado por el usuario."""
        if not self.current_circuit:
            print(f"{Colors.WARNING}⚠️ No hay ningún circuito para ejecutar. Usa `crear` y `agregar` primero.{Colors.ENDC}")
            return
        
        if self.current_circuit.size() == 0:
            print(f"{Colors.WARNING}⚠️ El circuito está vacío. Añade algunas puertas antes de ejecutar.{Colors.ENDC}")
            return
            
        try:
            print(f"\n{Colors.BOLD}🔧 Ejecutando tu circuito personalizado:{Colors.ENDC}")
            # Clonamos el circuito para no modificar el original al añadir medidas
            qc_to_run = self.current_circuit.copy()
            # Añadimos medidas a todos los qubits si no las tienen
            if qc_to_run.num_clbits == 0:
                qc_to_run.measure_all()
                print(f"{Colors.OKBLUE}ℹ️  Se han añadido mediciones a todos los qubits para la ejecución.{Colors.ENDC}")
            
            # Reutilizamos la lógica de ejecución
            self.execute_circuit(qc_to_run, backend_name, shots)

        except Exception as e:
            error_msg = f"❌ Error al ejecutar el circuito personalizado: {e}"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            self.logger.error(f"Error al ejecutar circuito personalizado: {e}", exc_info=True)
            return None

    def execute_circuit(self, qc: QuantumCircuit, backend_name: Optional[str] = None, shots: int = 1024):
        """Función genérica para ejecutar cualquier circuito."""
        self._initialize_service()
        
        print(qc.draw(output='text'))
        
        if backend_name:
            backend = self.service.backend(backend_name)
        else:
            print("🔎 No se especificó backend. Buscando el menos ocupado...")
            backend = self.service.least_busy(operational=True, simulator=False)
        
        print(f"🎯 Usando backend: {Colors.OKCYAN}{backend.name}{Colors.ENDC}")
        
        print(f"⚙️  Preparando ejecución con {shots} shots...")
        sampler = Sampler(backend)
        job = sampler.run([qc], shots=shots)
        job_id = job.job_id()
        
        print(f"🚀 Job enviado con ID: {Colors.OKCYAN}{job_id}{Colors.ENDC}")
        print("⏳ Esperando resultados (esto puede tardar)...")
        
        start_time = time.time()
        result = job.result()
        execution_time = time.time() - start_time
        
        print(f"\n{Colors.OKGREEN}✅ ¡Resultados obtenidos en {execution_time:.2f}s!{Colors.ENDC}")
        
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        print(f"{Colors.BOLD}📊 Resultados de la medición:{Colors.ENDC}")
        for outcome, count in counts.items():
            percentage = (count / shots) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {outcome}: {count:<5} ({percentage:5.2f}%) {Colors.OKBLUE}{bar}{Colors.ENDC}")

        self.logger.info(f"Job {job_id} completado exitosamente.")
        return job_id

    # --- FIN DE NUEVAS FUNCIONES ---


def print_interactive_help():
    """Muestra la ayuda para el modo interactivo."""
    print(f"\n{Colors.BOLD}--- Ayuda de Qiskit CLI Interactivo ---{Colors.ENDC}")
    print("Escribe un comando y presiona Enter. Algunos comandos aceptan argumentos adicionales.")
    
    # --- NUEVO: Ayuda actualizada con los comandos del constructor ---
    ayuda = {
        "--- Gestión de Cuenta y Backends ---": "",
        "configurar": "Guarda tu token de API de IBM Quantum.",
        "probar | conexion": "Verifica la conexión con los servicios de IBM Quantum.",
        "listar [--simuladores] [--todos]": "Muestra los backends. --simuladores solo muestra simuladores.",
        "mejor [--simulador]": "Encuentra el computador cuántico o simulador menos ocupado.",
        "estado <job_id>": "Comprueba el estado de un trabajo enviado.",
        "\n--- Constructor de Circuitos ---": "",
        "crear <num_qubits>": "Inicia un nuevo circuito con un número de qubits.",
        "agregar <puerta> <qubits...> [params...]": "Añade una puerta al circuito. Ej: `agregar h 0`, `agregar cx 0 1`, `agregar rx pi/2 0`.",
        "ver": "Muestra el diagrama del circuito actual.",
        "ejecutar_custom [--backend <nombre>] [--shots <num>]": "Ejecuta el circuito que has creado.",
        "limpiar_circuito": "Elimina el circuito actual para empezar de nuevo.",
        "\n--- Otros Comandos ---": "",
        "ejemplo [--backend <nombre>] [--shots <num>]": "Ejecuta un circuito de Bell de ejemplo (comando 'run' anterior).",
        "ayuda": "Muestra este mensaje de ayuda.",
        "limpiar | cls": "Limpia la pantalla de la consola.",
        "salir | exit | quit": "Sale de la aplicación."
    }

    for cmd, desc in ayuda.items():
        if desc == "":
            print(f"\n{Colors.OKCYAN}{cmd}{Colors.ENDC}")
        else:
            print(f"  {Colors.OKGREEN}{cmd:<45}{Colors.ENDC} {desc}")
    print("-" * 60)


def interactive_shell(cli: QiskitCLI):
    """Inicia el shell interactivo."""
    print_banner()
    while True:
        try:
            prompt = f"{Colors.OKGREEN}Qiskit-CLI{Colors.ENDC}{Colors.BOLD} > {Colors.ENDC}"
            user_input = input(prompt).strip()
            
            if not user_input:
                continue

            parts = user_input.split()
            command = parts[0].lower()
            args = parts[1:]

            if command in ["salir", "exit", "quit"]:
                print(f"{Colors.OKBLUE}¡Hasta la próxima aventura cuántica!{Colors.ENDC}")
                break
            
            elif command in ["ayuda", "help", "?"]:
                print_interactive_help()

            elif command in ["limpiar", "cls", "clear"]:
                os.system('cls' if os.name == 'nt' else 'clear')

            elif command == "configurar":
                token = input("🔑 Introduce tu token de API de IBM Quantum: ").strip()
                if token:
                    cli.setup_account(token)
                else:
                    print(f"{Colors.WARNING}El token no puede estar vacío.{Colors.ENDC}")
            
            elif command in ["probar", "conexion", "test"]:
                cli.test_connection()

            elif command in ["listar", "backends"]:
                is_op = "--todos" not in args
                is_sim = "--simuladores" in args if "--simuladores" in args else None
                cli.list_backends(operational=is_op, simulator=is_sim)

            elif command in ["mejor", "least-busy"]:
                is_sim = "--simulador" in args
                cli.get_least_busy_backend(simulator=is_sim)

            # --- NUEVO: Lógica para los nuevos comandos en el shell ---
            elif command == "crear":
                if not args:
                    print(f"{Colors.WARNING}Debes especificar el número de qubits. Uso: `crear <num_qubits>`{Colors.ENDC}")
                else:
                    cli.create_circuit(args[0])

            elif command == "agregar":
                cli.add_gate_to_circuit(args)

            elif command == "ver":
                cli.view_circuit()

            elif command == "limpiar_circuito":
                cli.clear_circuit()

            elif command in ["ejecutar_custom", "run_custom"]:
                backend_name = args[args.index("--backend") + 1] if "--backend" in args else None
                shots = int(args[args.index("--shots") + 1]) if "--shots" in args else 1024
                cli.run_custom_circuit(backend_name, shots)
            # --- FIN DE LA NUEVA LÓGICA ---

            elif command in ["ejemplo", "run"]: # Renombrado `run` a `ejemplo` para evitar confusión
                backend_name = args[args.index("--backend") + 1] if "--backend" in args else None
                shots = int(args[args.index("--shots") + 1]) if "--shots" in args else 1024
                cli.run_example_circuit(backend_name, shots)

            elif command in ["estado", "status"]:
                if not args:
                    print(f"{Colors.WARNING}Debes proporcionar un ID de trabajo. Uso: estado <job_id>{Colors.ENDC}")
                else:
                    cli.check_job_status(args[0])

            else:
                print(f"{Colors.FAIL}Comando desconocido: '{command}'. Escribe 'ayuda' para ver la lista de comandos.{Colors.ENDC}")

        except KeyboardInterrupt:
            print(f"\n{Colors.OKBLUE}Operación cancelada. Escribe 'salir' para terminar.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Ha ocurrido un error inesperado: {e}{Colors.ENDC}")
            cli.logger.error(f"Error en el shell interactivo con input '{user_input}': {e}", exc_info=True)


def main():
    # El resto del código `main` para manejar el modo no interactivo sigue igual
    # y no se ha modificado, ya que las nuevas funciones son para el modo interactivo.
    try:
        log_level = 'INFO'
        log_file = None
        if '--log-level' in sys.argv:
            log_level = sys.argv[sys.argv.index('--log-level') + 1]
        if '--log-file' in sys.argv:
            log_file = sys.argv[sys.argv.index('--log-file') + 1]
        logger = setup_logging(log_level, log_file)
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error fatal al configurar logging: {e}{Colors.ENDC}")
        sys.exit(1)

    cli = QiskitCLI(logger)

    if len(sys.argv) > 1 and not (len(sys.argv) == 2 and (sys.argv[1].startswith('--log'))):
        logger.info("Iniciando en modo no interactivo (argumentos detectados).")
        # El código del parser no interactivo original iría aquí...
        print("El modo no interactivo no soporta el constructor de circuitos. Iniciando en modo interactivo.")
        interactive_shell(cli)
    else:
        logger.info("Iniciando en modo interactivo.")
        interactive_shell(cli)


if __name__ == '__main__':
    main()
