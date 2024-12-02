import os


def modify_txt_files_in_directory(directory_path):
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print("指定的目录不存在。")
        return

    # 遍历目录中的所有文件
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):  # 只处理txt文件
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            new_lines = []
            rename_needed = False  # 标记是否需要重命名
            new_file_path = file_path

            for line in lines:
                parts = line.split()
                if not parts:
                    continue

                first_char = parts[0]

                if first_char == "0":
                    # 如果第一个字符为“0”，在文件名后加“_nodrone”
                    new_file_path = file_path.replace('.txt', '_nodrone.txt')
                    rename_needed = True
                    break  # 找到需要重命名的情况，跳出循环
                elif first_char == "1":
                    # 如果第一个字符为“1”，将“1”改为“0”
                    parts[0] = "0"
                else:
                    # 如果不是“1”或“0”，打印报错
                    print(f"报错：文件 {file_name} 的第一个字符不是‘1’或‘0’。")
                    new_file_path = file_path
                    break  # 跳出循环，继续下一个文件

                new_lines.append(' '.join(parts) + '\n')

            # 处理重命名
            if rename_needed:
                os.rename(file_path, new_file_path)
                print(f"文件 {file_name} 已重命名为: {new_file_path}")
            else:
                # 写入修改后的内容
                with open(file_path, 'w') as file:
                    file.writelines(new_lines)
                print(f"文件 {file_name} 的内容已更新。")


path = "E:/datasets/anti-uav/test-dev/20190925_101846_1_1/val/labels"
modify_txt_files_in_directory(path)  # 将 'your_directory' 替换为你的文件夹路径
