# MAML

## Training
URDF files are saved in urdf directory, which specify robot(number of components) of environment.
If you want to change the task of experiments, modify `--config` setting.

```bash
python main.py --config <config yml file>

python main.py --config 5way1shot.yml
nohup python main.py --config cifar10.yml > 5way1shot.out &
```
