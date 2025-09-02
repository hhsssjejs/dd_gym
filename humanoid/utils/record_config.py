import os
from humanoid import LEGGED_GYM_ENVS_DIR, LEGGED_GYM_ROOT_DIR


def record_config(log_root, urdf_path, name="a1_amp"):
    log_dir = os.path.join(log_root, "src")
    os.makedirs(log_dir, exist_ok=True)

    str_config = name + '_config.py'
    file_path1 = os.path.join(log_dir, str_config)
    file_path2 = os.path.join(log_dir, 'legged_robot_config.py')
    file_path5 = os.path.join(log_dir, 'legged_robot.py')
    str_urdf = urdf_path.split('/')[-1]
    file_path3 = os.path.join(log_dir, str_urdf)

    str_config1 = name + '_env.py'
    file_path4 = os.path.join(log_dir, str_config1)

    root1 = name.split('_')[0]

    root_path1 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, name + '_config.py')
    root_path2 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'legged_robot_config.py')
    root_path3 = urdf_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    root_path4 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, name + '_env.py')
    root_path5 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'legged_robot.py')

    # with open(root_path1, 'r', encoding='utf-8') as file:
    #     content = file.read()

    # with open(file_path1, 'w', encoding='utf-8') as file:
    #     file.write(content)

    # with open(root_path2, 'r',encoding='utf-8') as file:
    #     content = file.read()

    # with open(file_path2, 'w', encoding='utf-8') as file:
    #     file.write(content)

    # with open(root_path3, 'r',encoding='utf-8') as file:
    #     content = file.read()

    # with open(file_path3, 'w', encoding='utf-8') as file:
    #     file.write(content)

    # if os.path.exists(root_path4):
    #     with open(root_path4, 'r',encoding='utf-8') as file:
    #         content = file.read()

    #     with open(file_path4, 'w', encoding='utf-8') as file:
    #         file.write(content)

    # with open(root_path5, 'r',encoding='utf-8') as file:
    #     content = file.read()

    # with open(file_path5, 'w', encoding='utf-8') as file:
    #     file.write(content)

    root_paths = [root_path1, root_path2, root_path3, root_path4, root_path5]
    file_paths = [file_path1, file_path2, file_path3, file_path4, file_path5]

    for root_path, file_path in zip(root_paths, file_paths):
        if os.path.exists(root_path):
            with open(root_path, 'r', encoding='utf-8') as file:
                content = file.read()

            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
