## Preliminary work

1. Add Environment Variable
   1. Put `export PYTHONPATH="Your_path_to_repo/layout_design"` to your `.bashrc` or `.zshrc` file.
   2. `source ~/.bashrc` or `source ~/.zshrc`
2. Simplify the object file, reduce the number of vertices and faces.
   1. `cd ./dataset`
   2. `python simplify_obj.py `
   3. Credit to [Fast-Quadric-Mesh-Simplification](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification)

## Useful websites
1. URDF visualization: https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/
2. Inverse Dynamics in Mujoco: https://www.danaukes.com/work-blog/2024-03-11-double-pendulum-inverse/
3. TRUMANS dataset: https://drive.google.com/drive/folders/1CPw_cQcQAP_wbMuQ89rREQmMtEMd-2Hz?usp=sharing
4. [Evolution Strategies](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/)