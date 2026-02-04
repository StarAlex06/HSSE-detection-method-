import sys
import platform

print("=" * 60)
print("ПРОВЕРКА СИСТЕМЫ")
print("=" * 60)

print(f"Python версия: {sys.version}")
print(f"Платформа: {platform.platform()}")
print(f"Архитектура: {platform.machine()}")

# Проверяем разрядность Python
import struct
print(f"Разрядность Python: {struct.calcsize('P') * 8} бит")

print("\nРекомендации:")
if sys.version_info >= (3, 13):
    print("⚠️  Вы используете Python 3.13 - это очень новая версия!")
    print("   Рекомендую установить Python 3.11 для лучшей совместимости")
else:
    print("✅ Версия Python подходящая")

print("\nДля установки:")
if sys.version_info >= (3, 13):
    print("Используйте: python quick_install.py")
else:
    print("Используйте: pip install -r requirements_py311.txt")