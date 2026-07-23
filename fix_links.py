import os

files_to_update = [
    "/home/marwan/.dev/vecore/README.md",
    "/home/marwan/.dev/vecore/docs/index.html",
    "/home/marwan/.dev/vecore/docs/docs/cuda.html",
    "/home/marwan/.dev/vecore/docs/docs/data.html",
    "/home/marwan/.dev/vecore/docs/docs/getting-started.html",
    "/home/marwan/.dev/vecore/docs/docs/nn.html",
    "/home/marwan/.dev/vecore/docs/docs/optim.html",
    "/home/marwan/.dev/vecore/docs/docs/sequential.html",
    "/home/marwan/.dev/vecore/docs/docs/tensor.html"
]

for filepath in files_to_update:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Replace the specific URLs
        content = content.replace("https://github.com/Marwan7042/Vecore.com.git", "https://github.com/Marwan7042/Vecore.git")
        content = content.replace("https://github.com/Marwan7042/Vecore.com", "https://github.com/Marwan7042/Vecore")
        
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated {filepath}")
    else:
        print(f"File not found: {filepath}")
