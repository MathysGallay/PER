# ...existing code...
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = {
    "1": ("Llama 3.2 1B", "meta-llama/Llama-3.2-1B-Instruct", "llama-3.2-1b"),
    "2": ("Phi-3 Mini", "microsoft/Phi-3-mini-4k-instruct", "phi-3-mini"),
    "3": ("Gemma 2B", "google/gemma-2b-it", "gemma-2b"),
}

def main():
    print("Choisir un modèle :")
    for k, (name, repo, _) in MODELS.items():
        print(f"{k}. {name} ({repo})")
    choice = input("Votre choix (1/2/3): ").strip()

    if choice not in MODELS:
        print("Choix invalide.")
        return

    name, repo_id, folder = MODELS[choice]
    output_dir = Path("./models") / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Téléchargement de {name}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.md", ".gitattributes"]
    )
    print(f"Téléchargé dans: {output_dir}")

if __name__ == "__main__":
    main()