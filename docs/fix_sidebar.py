import os
import re

html_files = [
    "cuda.html", "data.html", "getting-started.html",
    "nn.html", "optim.html", "sequential.html", "tensor.html"
]

navbar_template = """<nav class="navbar">
    <div class="nav-content">
        <a href="../index.html" class="logo">Vecore</a>
        <div class="nav-links">
            <a href="../index.html">Home</a>
            <a href="../index.html#demo">Demo</a>
            <a href="getting-started.html" class="active">Docs</a>
            <a href="#">GitHub</a>
        </div>
    </div>
</nav>"""

sidebar_template = """<aside class="sidebar">
    <div class="sidebar-section">
        <ul class="sidebar-nav">
            <li><a href="getting-started.html">Getting Started</a></li>
        </ul>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-heading">Core API</div>
        <ul class="sidebar-nav">
            <li><a href="tensor.html">Tensor</a></li>
            <li><a href="nn.html">Dense Layer</a></li>
            <li><a href="sequential.html">Sequential</a></li>
        </ul>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-heading">Training</div>
        <ul class="sidebar-nav">
            <li><a href="optim.html">Optimizers</a></li>
            <li><a href="data.html">Data Loading</a></li>
        </ul>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-heading">Advanced</div>
        <ul class="sidebar-nav">
            <li><a href="cuda.html">CUDA Acceleration</a></li>
        </ul>
    </div>
</aside>"""

base_dir = "/home/marwan/.dev/vecore/docs/docs/"

for filename in html_files:
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract the main content (anything between <main...> and </main>, or if not found, just leave it alone)
    main_match = re.search(r'<main[^>]*>(.*?)</main>', content, re.DOTALL)
    if main_match:
        main_content = main_match.group(1)
        
        # Build the correct full body
        new_body = f"""<body>
    {navbar_template}
    <div class="layout-container">
        {sidebar_template}
        <main class="content">
{main_content}
        </main>
    </div>
</body>"""
        
        # Replace the entire body
        content = re.sub(r'<body.*?>.*?</body>', new_body, content, flags=re.DOTALL)
        
        # Set active link
        target = f'href="{filename}"'
        replacement = f'href="{filename}" class="active"'
        content = content.replace(target, replacement)
        
        with open(filepath, 'w') as f:
            f.write(content)

print("HTML files normalized.")
