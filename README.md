1. Add Environment Variable
   1. Put `export PYTHONPATH="Your_path_to_repo/layout_design"` to your `.bashrc` or `.zshrc` file.
   2. `source ~/.bashrc` or `source ~/.zshrc`
2. Simplify the object file, reduce the number of vertices and faces.
   1. `cd ./dataset`
   2. `python simplify_obj.py `
   3. Credit to [Fast-Quadric-Mesh-Simplification](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification)