#!/usr/bin/env python3#!/usr/bin/env python3
"""
CLI-Qis-Kit - Punto de entrada principal
by SmokAppSoftware jako with Claude AI, Gemini AI, COPILOT
Versión 2.3 - Constructor + Química + Neurona + Premium

Punto de entrada inteligente que detecta automáticamente las dependencias disponibles
y ejecuta la versión apropiada de la CLI.
"""

import sys
import os
from pathlib import Path

# Colores para output
class Colors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    OKCYAN = '\033[96m'

def print_banner():
    """Banner con el logo CLI-Qis-Kit"""
    print(f"""
{Colors.OKCYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██████╗██╗     ██╗      ██████╗ ██╗███████╗      ██╗  ██╗██╗████████╗
║  ██╔════╝██║     ██║     ██╔═══██╗██║██╔════╝      ██║ ██╔╝██║╚══██╔══╝
║  ██║     ██║     ██║     ██║   ██║██║███████╗█████╗█████╔╝ ██║   ██║   
║  ██║     ██║     ██║     ██║▄▄ ██║██║╚════██║╚════╝██╔═██╗ ██║   ██║   
║  ╚██████╗███████╗██║     ╚██████╔╝██║███████║      ██║  ██╗██║   ██║   
║   ╚═════╝╚══════╝╚═╝      ╚══▀▀═╝ ╚═╝╚══════╝      ╚═╝  ╚═╝╚═╝   ╚═╝   
║                                                                  ║
║               🚀 Ultimate Quantum Computing CLI                  ║
║            by SmokAppSoftware jako with AI Assistance           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝{Colors.ENDC}

{Colors.BOLD}⚡ CLI-Qis-Kit v2.3 - Quantum Computing Made Simple
🎯 Detectando capacidades del sistema...{Colors.ENDC}
""")

def check_dependencies():
    """Verifica qué dependencias están disponibles"""
    deps_status = {
        'qiskit': False,
        'qiskit_aer': False,
        'qiskit_ibm_runtime': False,
        'matplotlib': False,
        'numpy': False,
        'scipy': False
    }
    
    print("🔍 Verificando dependencias...")
    print("-" * 40)
    
    # Verificar dependencias básicas
    try:
        import qiskit
        deps_status['qiskit'] = True
        print(f"✅ Qiskit {qiskit.__version__}")
    except ImportError:
        print("❌ Qiskit no encontrado")
        return deps_status
    
    try:
        import qiskit_aer
        deps_status['qiskit_aer'] = True
        print(f"✅ Qiskit Aer {qiskit_aer.__version__}")
    except ImportError:
        print("⚠️  Qiskit Aer no encontrado - sin simulador local")
    
    try:
        import qiskit_ibm_runtime
        deps_status['qiskit_ibm_runtime'] = True
        print(f"✅ IBM Quantum Runtime {qiskit_ibm_runtime.__version__}")
    except ImportError:
        print("⚠️  IBM Quantum Runtime no encontrado - solo simulador local")
    
    try:
        import matplotlib
        deps_status['matplotlib'] = True
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("⚠️  Matplotlib no encontrado - sin gráficos")
    
    try:
        import numpy
        deps_status['numpy'] = True
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError:
        print("⚠️  NumPy no encontrado")
    
    try:
        import scipy
        deps_status['scipy'] = True
        print(f"✅ SciPy {scipy.__version__}")
    except ImportError:
        print("⚠️  SciPy no encontrado - funcionalidades limitadas")
    
    return deps_status

def check_extensions():
    """Verifica si las extensiones cuánticas están disponibles"""
    try:
        import numpy
        import matplotlib
        # Verificar si existe el archivo de extensiones
        extensions_file = Path(__file__).parent / "quantum_extensions.py"
        cli_extended_file = Path(__file__).parent / "qiskit_cli_extended.py"
        
        if extensions_file.exists() and cli_extended_file.exists():
            print("✅ Extensiones cuánticas encontradas (Química + Neurona)")
            return True
        else:
            print("⚠️  Archivos de extensiones no encontrados")
            return False
    except ImportError as e:
        print(f"⚠️  Dependencias de extensiones faltantes: {e}")
        return False

def check_premium_system():
    """Verifica si el sistema premium está disponible"""
    try:
        license_file = Path(__file__).parent / "license_manager.py"
        premium_file = Path(__file__).parent / "premium_features.py"
        
        if license_file.exists() and premium_file.exists():
            print("✅ Sistema Premium disponible")
            return True
        else:
            print("⚠️  Sistema Premium no encontrado")
            return False
    except Exception:
        return False

def print_installation_guide():
    """Muestra la guía de instalación"""
    print("\n" + "="*60)
    print("📦 GUÍA DE INSTALACIÓN")
    print("="*60)
    print("\n🔧 Para funcionalidad completa, instala todas las dependencias:")
    print("\n   pip install -r requirements.txt")
    print("\n📋 O instala manualmente:")
    print("   pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib numpy scipy")
    print("\n💡 Para uso básico (solo simulador local):")
    print("   pip install qiskit qiskit-aer matplotlib numpy")
    print("\n🌐 Para usar hardware cuántico real de IBM:")
    print("   pip install qiskit-ibm-runtime")
    print("   • Regístrate en: https://quantum.ibm.com/")
    print("   • Obtén tu API token del dashboard")
    print("   • Usa el comando 'configurar' en la CLI")
    print("\n🌟 PREMIUM: Para química cuántica avanzada y ML cuántico:")
    print("   • Activa con clave demo: DEMO-PREMIUM-2024")
    print("   • Más info: https://github.com/smokappstore/CLI-Qis-kit-")
    print("\n" + "="*60)

def select_cli_version(deps_status):
    """Selecciona qué versión de la CLI ejecutar basada en las dependencias"""
    
    if not deps_status['qiskit']:
        print("\n❌ ERROR: Qiskit es requerido para ejecutar la CLI.")
        print("   Instala con: pip install qiskit")
        return None
    
    # Verificar si hay extensiones disponibles
    extensions_available = check_extensions()
    premium_available = check_premium_system()
    
    # Prioridad: Extended > Premium > Complete > Original
    if extensions_available:
        print("\n🌟 Iniciando CLI EXTENDIDA (Constructor + Química + Neurona Cuántica)")
        if premium_available:
            print("   🎯 Sistema Premium detectado - Funcionalidades avanzadas disponibles")
        return 'qiskit_cli_extended.py'
    
    # Si tiene todo básico, usar la versión completa normal
    elif deps_status['qiskit_aer'] and deps_status['matplotlib'] and deps_status['numpy']:
        if deps_status['qiskit_ibm_runtime']:
            print("\n🚀 Iniciando CLI completa (Simulador Local + IBM Quantum)")
            return 'run_cli.py'
        else:
            print("\n🖥️  Iniciando CLI con simulador local únicamente")
            return 'run_cli.py'
    
    # Si no tiene aer pero sí IBM runtime, usar versión original
    elif deps_status['qiskit_ibm_runtime']:
        print("\n☁️  Iniciando CLI solo con IBM Quantum (sin simulador local)")
        return 'qiskit_cli.py'
    
    else:
        print("\n❌ ERROR: Se requiere al menos Qiskit Aer O IBM Runtime para ejecutar la CLI.")
        print_installation_guide()
        return None

def show_welcome_message():
    """Muestra mensaje de bienvenida con tips"""
    print(f"\n{Colors.BOLD}🎉 ¡Bienvenido a CLI-Qis-Kit!{Colors.ENDC}")
    print("\n💡 Primeros pasos:")
    print("   • Escribe 'ayuda' para ver todos los comandos")
    print("   • Prueba 'demo' para circuitos de ejemplo")
    print("   • Usa 'crear 2' para tu primer circuito cuántico")
    
    # Tips específicos según capacidades
    extensions_available = check_extensions()
    if extensions_available:
        print("\n🧪 Funciones especiales disponibles:")
        print("   • 'quimica' - Tabla periódica cuántica")
        print("   • 'neurona' - Redes neuronales cuánticas")
        print("   • 'demo_quimica' - Demo de química cuántica")
    
    premium_available = check_premium_system()
    if premium_available:
        print("\n🌟 Sistema Premium:")
        print("   • Usa 'licencia DEMO-PREMIUM-2024' para prueba gratis")
        print("   • 50+ elementos químicos y moléculas complejas")
    
    print(f"\n{Colors.OKCYAN}═══════════════════════════════════════════════{Colors.ENDC}")

def main():
    """Función principal"""
    print_banner()
    
    try:
        # Verificar dependencias
        deps_status = check_dependencies()
        
        print("-" * 40)
        selected_cli = select_cli_version(deps_status)
        
        if not selected_cli:
            print(f"\n{Colors.FAIL}❌ No se puede ejecutar CLI-Qis-Kit.{Colors.ENDC}")
            print_installation_guide()
            sys.exit(1)
        
        # Verificar que el archivo CLI existe
        cli_path = Path(__file__).parent / selected_cli
        if not cli_path.exists():
            print(f"\n❌ ERROR: Archivo {selected_cli} no encontrado.")
            print(f"   Asegúrate de que esté en el mismo directorio que main.py")
            print(f"   Archivos disponibles: {list(Path(__file__).parent.glob('*.py'))}")
            sys.exit(1)
        
        # Mostrar mensaje de bienvenida
        show_welcome_message()
        
        print(f"\n🎯 Ejecutando: {selected_cli}")
        print("=" * 60 + "\n")
        
        # Importar y ejecutar la CLI seleccionada
        if selected_cli == 'qiskit_cli_extended.py':
            # Importar y ejecutar la versión extendida
            import qiskit_cli_extended
            qiskit_cli_extended.main()
        elif selected_cli == 'run_cli.py':
            # Importar y ejecutar la versión con simulador local
            import run_cli
            run_cli.main()
        else:
            # Importar y ejecutar la versión original
            import qiskit_cli
            qiskit_cli.main()
            
    except KeyboardInterrupt:
        print(f"\n{Colors.OKCYAN}👋 ¡Gracias por usar CLI-Qis-Kit!{Colors.ENDC}")
        print("🌟 Si te gustó, considera darle una estrella en GitHub")
        print("💰 Support: https://github.com/sponsors/smokappstore")
    except ImportError as e:
        print(f"\n{Colors.FAIL}❌ Error de importación: {e}{Colors.ENDC}")
        print("\n🔧 Solución:")
        print("1. Verifica que todos los archivos estén en el directorio")
        print("2. Instala dependencias: pip install -r requirements.txt")
        print("3. Si persiste, reporta el error en GitHub")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Error inesperado: {e}{Colors.ENDC}")
        print("\n📞 Para soporte:")
        print("   GitHub: https://github.com/smokappstore/CLI-Qis-kit-")
        print("   Email: jakocrazykings@gmail.com")
        sys.exit(1)

if __name__ == "__main__":
    main()
