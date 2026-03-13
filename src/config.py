import yaml
from pathlib import Path
from easydict import EasyDict as edict

project_dir = Path(__file__).resolve().parent.parent


def get_config(config_path: str = f"{project_dir}/config.yml") -> edict:
    """
    加载 YAML 配置文件并转换为支持点号访问的 EasyDict 对象。
    :param config_path: 配置文件路径
    :return: 配置对象，可通过 config.model.lr 方式访问
    """

    if not Path(config_path).exists():
        raise FileNotFoundError(f"配置文件未找到： {config_path}")

    with open(config_path, "r") as f:
        cfg = edict(yaml.safe_load(f))

    for k, v in cfg['path'].items():
        abs_path = (project_dir / v).resolve()
        cfg['path'][k] = abs_path

    return cfg

cfg = get_config()


# 辅助功能：动态修改配置（可选，常用于命令行覆盖）
def update_config(cfg, key_list, value):
    """
    通过键列表更新嵌套配置，例如 update_config(cfg, ['train', 'lr'], 0.001)
    """
    for key in key_list[:-1]:
        cfg = cfg[key]

    cfg[key_list[-1]] = value


if __name__ == '__main__':
    cfg = get_config()

    print(cfg.path.raw_data)

    update_config(cfg, ['train', 'epoch'], 8)

    print(cfg.train.epoch)

