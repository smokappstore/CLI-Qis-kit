#!/usr/bin/env python3
"""
Instalador Inteligente para CLI-Qis-Kit
by SmokAppSoftware jako with Claude AI, Gemini AI, COPILOT
Versión 2.3 - Instalador Automático con Detección de Sistema

Características:
- Detección automática de Python y pip
- Instalación por niveles (mínima, completa, desarrollo)
- Manejo de errores y reinstalación
- Verificación post-instalación
- Soporte para múltiples sistemas operativos
"""

import sys
import os
import subprocess
import platform
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

# Colores para la consola
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

class SystemInfo:
    """Clase para obtener información del sistema"""
    
    @staticmethod
    def get_python_info() -> Dict:
        """Obtiene información de Python"""
        return {
            'version': sys.version,
            'version_info': sys.version_info,
            'executable': sys.executable,
            'platform': platform.platform(),
            'architecture': platform.architecture()
        }
    
    @staticmethod
    def get_pip_command() -> str:
        """Determina el comando pip correcto para usar"""
        commands_to_try = [
            f"{sys.executable} -m pip",
            "pip3",
            "pip"
        ]
        
        for cmd in commands_to_try:
            try:
                result = subprocess.run(
                    cmd.split() + ["--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    return cmd
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        return "pip"  # Fallback
    
    @staticmethod
    def check_internet_connection() -> bool:
        """Verifica si hay conexión a internet"""
        try:
            import urllib.request
            urllib.request.urlopen('https://pypi.org', timeout=5)
            return True
        except:
            return False

class QiskitInstaller:
    """Instalador principal para CLI-Qis-Kit"""
    
    def __init__(self):
        self.system_info = SystemInfo()
        self.pip_command = self.system_info.get_pip_command()
        self.install_log = []
        self.errors = []
        
        # Configuraciones de instalación
        self.install_configs = {
            'minimal': {
                'name': 'Instalación Mínima',
                'description': 'Solo simulador local - Sin conexión IBM',
                'packages': [
                    'qiskit>=0.45.0',
                    'qiskit-aer>=0.12.0',
                    'matplotlib>=3.5.0',
                    'numpy>=1.21.0',
                    'Pillow>=9.0.0'
                ],
                'optional': []
            },
            'complete': {
                'name': 'Instalación Completa',
                'description': 'Simulador local + IBM Quantum + Extensiones',
                'packages': [
                    'qiskit>=0.45.0',
                    'qiskit-aer>=0.12.0',
                    'qiskit-ibm-runtime>=0.15.0',
                    'matplotlib>=3.5.0',
                    'numpy>=1.21.0',
                    'scipy>=1.9.0',
                    'Pillow>=9.0.0'
                ],
                'optional': []
            },
            'development': {
                'name': 'Instalación para Desarrollo',
                'description': 'Todo + herramientas de desarrollo',
                'packages': [
                    'qiskit>=0.45.0',
                    'qiskit-aer>=0.12.0',
                    'qiskit-ibm-runtime>=0.15.0',
                    'matplotlib>=3.5.0',
                    'numpy>=1.21.0',
                    'scipy>=1.9.0',
                    'Pillow>=9.0.0'
                ],
                'optional': [
                    'pytest>=7.0.0',
                    'pytest-cov>=4.0.0',
                    'black>=22.0.0',
                    'flake8>=5.0.0',
                    'mypy>=1.0.0'
                ]
            }
        }
    
    def print_banner(self):
        """Imprime el banner del instalador"""
        banner = f"""
{Colors.OKCYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                                 ║
║   ██╗███╗   ██╗███████╗████████╗ █████╗ ██╗      █████╗ ██████╗  ██████╗ ███████╗║
║   ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██╔══██╗██╔══██╗██╔══ ██╗██╔══██╗║
║   ██║██╔██╗ ██║███████╗   ██║   ███████║██║     ███████║██║  ██║██╔   ██╗██████╗ ║
║   ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██╔══██║██║  ██║██╔   ██╗██╔  ██╗║
║   ██║██║ ╚████║███████║   ██║   ██║  ██║███████╗██║  ██║██████╔╝ ██████╗ ██╔  ██╗║
║   ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝  ╚══════╝╚═╝  ╚═╝║
║                                                                                 ║
║               🚀 CLI-Qis-Kit Instalador Automático                              ║
║            by SmokAppSoftware jako with AI Assistance                           ║
║                                                                   ______________║
╚══════════════════════════════════════════════════════════════════╝{Colors.ENDC}

{Colors.BOLD}⚡ Versión: 2.3 - Instalador Inteligente
📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Detectando configuración del sistema...{Colors.ENDC}
"""
        print(banner)
    
    def log_action(self, message: str, level: str = "info"):
        """Registra acciones en el log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {level.upper()}: {message}"
        self.install_log.append(log_entry)
        
        if level == "error":
            self.errors.append(message)
        
        # También imprimir en consola para feedback inmediato
        color = Colors.OKGREEN if level == "info" else Colors.FAIL if level == "error" else Colors.WARNING
        print(f"{color}[{timestamp}] {message}{Colors.ENDC}")
    
    def check_system_requirements(self) -> bool:
        """Verifica los requisitos del sistema"""
        print(f"\n{Colors.BOLD}🔍 Verificando Requisitos del Sistema{Colors.ENDC}")
        print("-" * 50)
        
        # Verificar Python
        py_info = self.system_info.get_python_info()
        python_version = py_info['version_info']
        
        print(f"🐍 Python: {py_info['version']}")
        print(f"💻 Sistema: {py_info['platform']}")
        print(f"🏗️  Arquitectura: {py_info['architecture'][0]}")
        
        if python_version < (3, 8):
            self.log_action(f"Python {python_version} no soportado. Mínimo: Python 3.8", "error")
            return False
        
        # Verificar pip
        print(f"📦 Comando pip: {self.pip_command}")
        
        # Verificar internet
        has_internet = self.system_info.check_internet_connection()
        internet_status = "✅ Conectado" if has_internet else "❌ Sin conexión"
        print(f"🌐 Internet: {internet_status}")
        
        if not has_internet:
            self.log_action("Sin conexión a internet. Instalación puede fallar.", "warning")
        
        self.log_action("Verificación de sistema completada", "info")
        return True
    
    def run_command(self, command: str, description: str = "") -> bool:
        """Ejecuta un comando y maneja errores"""
        try:
            self.log_action(f"Ejecutando: {description or command}", "info")
            
            process = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos de timeout
            )
            
            if process.returncode == 0:
                self.log_action(f"✅ Éxito: {description or command}", "info")
                return True
            else:
                error_msg = process.stderr or process.stdout or "Error desconocido"
                self.log_action(f"❌ Fallo: {description or command} - {error_msg}", "error")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_action(f"❌ Timeout: {description or command}", "error")
            return False
        except Exception as e:
            self.log_action(f"❌ Excepción: {description or command} - {str(e)}", "error")
            return False
    
    def install_package_list(self, packages: List[str], description: str = "paquetes") -> bool:
        """Instala una lista de paquetes"""
        self.log_action(f"Iniciando instalación de {len(packages)} {description}", "info")
        
        success_count = 0
        for package in packages:
            print(f"\n📦 Instalando {package}...")
            
            # Intentar instalación con upgrade
            install_cmd = f"{self.pip_command} install --upgrade {package}"
            
            if self.run_command(install_cmd, f"instalación de {package}"):
                success_count += 1
                print(f"{Colors.OKGREEN}✅ {package} instalado exitosamente{Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}❌ Error instalando {package}{Colors.ENDC}")
                
                # Intentar sin --upgrade como fallback
                fallback_cmd = f"{self.pip_command} install {package}"
                if self.run_command(fallback_cmd, f"instalación fallback de {package}"):
                    success_count += 1
                    print(f"{Colors.WARNING}⚠️ {package} instalado con fallback{Colors.ENDC}")
        
        success_rate = (success_count / len(packages)) * 100
        self.log_action(f"Instalación completada: {success_count}/{len(packages)} ({success_rate:.1f}%)", "info")
        
        return success_count == len(packages)
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verifica que los paquetes estén instalados correctamente"""
        print(f"\n{Colors.BOLD}🔍 Verificando Instalación{Colors.ENDC}")
        print("-" * 40)
        
        test_imports = {
            'qiskit': 'qiskit',
            'qiskit_aer': 'qiskit-aer',
            'qiskit_ibm_runtime': 'qiskit-ibm-runtime',
            'matplotlib': 'matplotlib',
            'numpy': 'numpy',
            'scipy': 'scipy'
        }
        
        results = {}
        
        for module_name, package_name in test_imports.items():
            try:
                __import__(module_name)
                print(f"✅ {package_name}")
                results[package_name] = True
            except ImportError:
                print(f"❌ {package_name}")
                results[package_name] = False
        
        return results
    
    def show_installation_options(self) -> str:
        """Muestra opciones de instalación y permite al usuario elegir"""
        print(f"\n{Colors.BOLD}📋 Opciones de Instalación Disponibles{Colors.ENDC}")
        print("=" * 60)
        
        for i, (key, config) in enumerate(self.install_configs.items(), 1):
            print(f"\n{Colors.OKCYAN}[{i}] {config['name']}{Colors.ENDC}")
            print(f"    {config['description']}")
            print(f"    Paquetes principales: {len(config['packages'])}")
            if config['optional']:
                print(f"    Paquetes opcionales: {len(config['optional'])}")
        
        print(f"\n{Colors.BOLD}[4] Instalación Personalizada{Colors.ENDC}")
        print("    Seleccionar paquetes manualmente")
        
        print(f"\n{Colors.BOLD}[5] Solo Verificación{Colors.ENDC}")
        print("    Verificar instalación existente")
        
        print("=" * 60)
        
        while True:
            try:
                choice = input(f"{Colors.BOLD}Selecciona una opción (1-5): {Colors.ENDC}").strip()
                
                if choice == "1":
                    return "minimal"
                elif choice == "2":
                    return "complete"
                elif choice == "3":
                    return "development"
                elif choice == "4":
                    return "custom"
                elif choice == "5":
                    return "verify_only"
                else:
                    print(f"{Colors.WARNING}Opción inválida. Introduce un número del 1 al 5.{Colors.ENDC}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.OKBLUE}Instalación cancelada por el usuario.{Colors.ENDC}")
                sys.exit(0)
    
    def custom_installation(self) -> bool:
        """Permite instalación personalizada de paquetes"""
        print(f"\n{Colors.BOLD}🛠️ Instalación Personalizada{Colors.ENDC}")
        print("Introduce los paquetes a instalar, separados por espacios:")
        print("Ejemplo: qiskit qiskit-aer matplotlib")
        
        try:
            packages_input = input(f"{Colors.OKCYAN}Paquetes: {Colors.ENDC}").strip()
            if not packages_input:
                print(f"{Colors.WARNING}No se introdujeron paquetes.{Colors.ENDC}")
                return False
            
            packages = packages_input.split()
            return self.install_package_list(packages, "paquetes personalizados")
            
        except KeyboardInterrupt:
            print(f"\n{Colors.OKBLUE}Instalación cancelada por el usuario.{Colors.ENDC}")
            return False
    
    def save_installation_report(self):
        """Guarda un reporte de la instalación"""
        report_dir = Path.home() / ".qiskit_cli"
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"install_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info.get_python_info(),
            'pip_command': self.pip_command,
            'install_log': self.install_log,
            'errors': self.errors,
            'verification_results': self.verify_installation()
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n{Colors.OKGREEN}📄 Reporte guardado en: {report_file}{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.WARNING}⚠️ No se pudo guardar el reporte: {e}{Colors.ENDC}")
    
    def show_post_install_info(self):
        """Muestra información post-instalación"""
        print(f"\n{Colors.BOLD}🎉 Instalación Completada{Colors.ENDC}")
        print("=" * 50)
        
        print(f"\n{Colors.OKCYAN}📚 Primeros Pasos:{Colors.ENDC}")
        print("1. Ejecuta la CLI: python main.py")
        print("2. Prueba el simulador: escribe 'demo' en la CLI")
        print("3. Crea tu primer circuito: 'crear 2' luego 'agregar h 0'")
        
        if any('ibm' in pkg for pkg in self.install_log if 'instalado' in pkg):
            print(f"\n{Colors.OKCYAN}🔐 Para usar IBM Quantum:{Colors.ENDC}")
            print("1. Regístrate en: https://quantum.ibm.com/")
            print("2. Obtén tu API token")
            print("3. En la CLI: 'configurar' y pega tu token")
        
        print(f"\n{Colors.OKCYAN}📖 Recursos Útiles:{Colors.ENDC}")
        print("• GitHub: https://github.com/smokappstore/CLI-Qis-kit-")
        print("• Documentación Qiskit: https://qiskit.org/documentation/")
        print("• Ayuda en la CLI: escribe 'ayuda'")
        
        if self.errors:
            print(f"\n{Colors.WARNING}⚠️ Se encontraron {len(self.errors)} errores durante la instalación.{Colors.ENDC}")
            print("Revisa el reporte de instalación para más detalles.")
    
    def run_installation(self):
        """Ejecuta el proceso completo de instalación"""
        self.print_banner()
        
        # Verificar requisitos
        if not self.check_system_requirements():
            print(f"\n{Colors.FAIL}❌ Los requisitos del sistema no se cumplen. Instalación abortada.{Colors.ENDC}")
            return False
        
        # Mostrar opciones y obtener selección
        choice = self.show_installation_options()
        
        success = False
        
        if choice == "verify_only":
            self.verify_installation()
            success = True
            
        elif choice == "custom":
            success = self.custom_installation()
            
        elif choice in self.install_configs:
            config = self.install_configs[choice]
            
            print(f"\n{Colors.BOLD}🚀 Iniciando {config['name']}{Colors.ENDC}")
            print(f"📝 {config['description']}")
            print("-" * 50)
            
            # Actualizar pip primero
            print(f"\n{Colors.OKCYAN}📦 Actualizando pip...{Colors.ENDC}")
            self.run_command(f"{self.pip_command} install --upgrade pip", "actualización de pip")
            
            # Instalar paquetes principales
            main_success = self.install_package_list(config['packages'], "principales")
            
            # Instalar paquetes opcionales si existen
            optional_success = True
            if config['optional']:
                print(f"\n{Colors.OKCYAN}🔧 Instalando paquetes opcionales...{Colors.ENDC}")
                optional_success = self.install_package_list(config['optional'], "opcionales")
            
            success = main_success and optional_success
        
        # Verificar instalación
        if success:
            verification_results = self.verify_installation()
            failed_packages = [pkg for pkg, result in verification_results.items() if not result]
            
            if failed_packages:
                print(f"\n{Colors.WARNING}⚠️ Algunos paquetes no se verificaron correctamente: {', '.join(failed_packages)}{Colors.ENDC}")
                success = False
        
        # Guardar reporte y mostrar información final
        self.save_installation_report()
        
        if success:
            self.show_post_install_info()
        else:
            print(f"\n{Colors.FAIL}❌ La instalación se completó con errores.{Colors.ENDC}")
            print("Revisa el reporte de instalación y los logs para más detalles.")
        
        return success


def main():
    """Función principal del instalador"""
    try:
        installer = QiskitInstaller()
        success = installer.run_installation()
        
        if success:
            print(f"\n{Colors.OKGREEN}🎊 ¡Instalación exitosa! CLI-Qis-Kit está listo para usar.{Colors.ENDC}")
            sys.exit(0)
        else:
            print(f"\n{Colors.FAIL}💥 Instalación completada con errores.{Colors.ENDC}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.OKBLUE}👋 Instalación cancelada por el usuario.{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.FAIL}💥 Error inesperado en el instalador: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
