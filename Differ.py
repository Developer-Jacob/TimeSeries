import numpy as np


# (100, 10, 10)
def diff_data(input_data):
    diffed_data = np.transpose(np.diff(np.transpose(input_data)))
    pre_input = input_data[:-1]
    data = np.where(pre_input != 0, diffed_data / pre_input, 0) * 100
    return data


def diff_target(target_data):
    deffed_target = np.diff(target_data)
    pre_input2 = target_data[..., :-1]
    target = np.where(pre_input2 != 0, deffed_target / pre_input2, 0) * 100
    return target

def restore_data(diffed_data, origin):
    restored_data = [origin[0]]  # 첫 번째 행을 기준으로 시작
    for i, diff_row in enumerate(diffed_data):
        # origin[i]를 기준으로 다음 행 복원
        restored_row = origin[i] * (1 + diff_row / 100)
        restored_data.append(restored_row)
    return np.array(restored_data)

def restore_target(diffed_target, origin):
    restored_target = [origin[0]]  # 첫 번째 값을 기준으로 시작
    for i, diff in enumerate(diffed_target):
        # origin[i]를 기준으로 다음 값 복원
        restored_value = origin[i] * (1 + diff / 100)
        restored_target.append(restored_value)
    return np.array(restored_target)


if __name__ == "__main__":
    data = np.array([[1,5,9], [3,3,4]])
    target = np.array([1,8,2,4])
    diffed_data = diff_data(data)
    diffed_target = diff_target(target)
    print(diffed_data)
    print(diffed_target)
    print(restore_data(diffed_data, data))
    print(restore_target(diffed_target, target))