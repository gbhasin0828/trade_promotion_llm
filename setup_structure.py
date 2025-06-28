import os
from pathlib import Path

def create_directory_structure():
    """Create the complete directory structure for the trade promotion optimizer."""
    
    base_dir = Path("trade_promotion_optimizer")
    
    # Define the directory structure
    directories = [
        "config",
        "data/loaders",
        "data/validators", 
        "data/preprocessors",
        "features",
        "models/demand",
        "models/profit",
        "models/optimization",
        "llm/engines",
        "llm/semantic",
        "llm/rag",
        "optimization/algorithms",
        "optimization/constraints",
        "optimization/objectives",
        "api/routes",
        "api/middleware",
        "api/schemas",
        "utils",
        "training",
        "tests/test_data",
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "scripts",
        "notebooks",
        "docker/requirements"
    ]
    
    # Create directories
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
        
        # Create __init__.py files for Python packages
        if not directory.startswith(('tests/test_data', 'notebooks', 'docker', 'scripts')):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"Created __init__.py: {init_file}")
    
    # Create essential files
    essential_files = [
        ".gitignore",
        "README.md", 
        "requirements.txt",
        ".env.example"
    ]
    
    for file_name in essential_files:
        file_path = base_dir / file_name
        if not file_path.exists():
            file_path.touch()
            print(f"Created file: {file_path}")
    
    print("\n✅ Directory structure created successfully!")
    print(f"📁 Root directory: {base_dir.absolute()}")

if __name__ == "__main__":
    create_directory_structure()