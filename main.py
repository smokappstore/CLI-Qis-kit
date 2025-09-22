#!/usr/bin/env python3
"""
Qiskit Runtime CLI - Punto de entrada principal
by SmokAppSoftware jako with Claude AI, Gemini AI, COPILOT
Versión 2.2 - Constructor de Circuitos Interactivo con Simulador Local

Punto de entrada principal que detecta automáticamente las dependencias disponibles
y ejecuta la versión apropiada de la CLI.
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Verifica qué dependencias están disponibles"""
    deps_status = {
        'qiskit': False,
        'qiskit_aer': False,
        'qiskit_ibm_runtime': False,
        'matplotlib': False,
        'numpy': False
    }
    
    # Verificar dependencias básicas
    try:
        import qiskit
        deps_status['qiskit'] = True
        print("✅ Qiskit encontrado")
    except ImportError:
        print("❌ Qiskit no encontrado")
        return deps_status
    
    try:
        import qiskit_aer
        deps_status['qiskit_aer'] = True
        print("✅ Qiskit Aer (simulador) encontrado")
    except ImportError:
        print("⚠️  Qiskit Aer no encontrado - sin simulador local")
    
    try:
        import qiskit_ibm_runtime
        deps_status['qiskit_ibm_runtime'] = True
        print("✅ IBM Quantum Runtime encontrado")
    except ImportError:
        print("⚠️  IBM Quantum Runtime no encontrado - solo simulador local")
    
    try:
        import matplotlib
        deps_status['matplotlib'] = True
        print("✅ Matplotlib encontrado")
    except ImportError:
        print("⚠️  Matplotlib no encontrado - sin gráficos")
    
    try:
        import numpy
        deps_status['numpy'] = True
        print("✅ NumPy encontrado")
    except ImportError:
        print("⚠️  NumPy no encontrado")
    
    return deps_status

def print_installation_guide():
    """Muestra la guía de instalación"""
    print("\n" + "="*60)
    print("📦 GUÍA DE INSTALACIÓN")
    print("="*60)
    print("\n🔧 Para funcionalidad completa, instala todas las dependencias:")
    print("\n   pip install -r requirements.txt")
    print("\n📋 O instala manualmente:")
    print("   pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib numpy")
    print("\n💡 Para uso básico (solo simulador local):")
    print("   pip install qiskit qiskit-aer matplotlib numpy")
    print("\n🌐 Para usar hardware cuántico real de IBM:")
    print("   pip install qiskit-ibm-runtime")
    print("   • Regístrate en: https://quantum.ibm.com/")
    print("   • Obtén tu API token del dashboard")
    print("   • Usa el comando 'configurar' en la CLI")
    print("\n" + "="*60)

def select_cli_version(deps_status):
    """Selecciona qué versión de la CLI ejecutar basada en las dependencias"""
    
    if not deps_status['qiskit']:
        print("\n❌ ERROR: Qiskit es requerido para ejecutar la CLI.")
        print("   Instala con: pip install qiskit")
        return None
    
    # Si tiene todo, usar la versión completa
    if deps_status['qiskit_aer'] and deps_status['matplotlib'] and deps_status['numpy']:
        if deps_status['qiskit_ibm_runtime']:
            print("\n🚀 Iniciando CLI completa (Simulador Local + IBM Quantum)")
            return 'run_cli.py'  # Versión con simulador local y IBM
        else:
            print("\n🖥️  Iniciando CLI con simulador local únicamente")
            return 'run_cli.py'  # Versión con simulador local
    
    # Si no tiene aer pero sí IBM runtime, usar versión original
    elif deps_status['qiskit_ibm_runtime']:
        print("\n☁️  Iniciando CLI solo con IBM Quantum (sin simulador local)")
        return 'qiskit_cli.py'  # Versión original solo IBM
    
    else:
        print("\n❌ ERROR: Se requiere al menos Qiskit Aer O IBM Runtime para ejecutar la CLI.")
        print_installation_guide()
        return None

def main():
    """Función principal"""
    print("🔍 Verificando dependencias...")
    print("-" * 40)
    
    deps_status = check_dependencies()
    
    print("-" * 40)
    selected_cli = select_cli_version(deps_status)
    
    if not selected_cli:
        sys.exit(1)
    
    # Verificar que el archivo CLI existe
    cli_path = Path(__file__).parent / selected_cli
    if not cli_path.exists():
        print(f"\n❌ ERROR: Archivo {selected_cli} no encontrado.")
        print(f"   Asegúrate de que esté en el mismo directorio que main.py")
        sys.exit(1)
    
    print(f"\n🎯 Ejecutando: {selected_cli}")
    print("=" * 60 + "\n")
    
    # Importar y ejecutar la CLI seleccionada
    try:
        if selected_cli == 'run_cli.py':
            # Importar y ejecutar la versión con simulador local
            import run_cli
            run_cli.main()
        else:
            # Importar y ejecutar la versión original
            import qiskit_cli
            qiskit_cli.main()
    except KeyboardInterrupt:
        print(f"\n👋 ¡CLI cerrada por el usuario!")
    except Exception as e:
        print(f"\n❌ Error al ejecutar la CLI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
