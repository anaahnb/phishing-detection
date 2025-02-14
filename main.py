import subprocess
import os

def criar_executavel():
    caminho_interface = "Interface"
    script_gui = os.path.join(caminho_interface, "gui.py")
    destino_executavel = os.path.join(caminho_interface, "dist")

    # Garante que o diretório de destino existe
    os.makedirs(destino_executavel, exist_ok=True)

    # Comando para criar o executável
    comando = [
        "pyinstaller", "--onefile", "--noconsole", "--distpath", destino_executavel, script_gui
    ]

    print("Criando executável...")
    subprocess.run(comando, check=True)
    print(f"Executável criado em: {destino_executavel}")

if __name__ == "__main__":
    criar_executavel()