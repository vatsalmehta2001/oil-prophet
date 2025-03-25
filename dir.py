import os

def generate_tree(start_path, output_file="directory_structure.txt"):
    with open(output_file, "w") as f:
        for root, dirs, files in os.walk(start_path):
            level = root.replace(start_path, "").count(os.sep)
            indent = "    " * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            sub_indent = "    " * (level + 1)
            for file in files:
                f.write(f"{sub_indent}{file}\n")

# Specify the root directory of your project
if __name__ == "__main__":
    project_dir = "."
    generate_tree(project_dir)
