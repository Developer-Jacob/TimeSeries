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


if __name__ == "__main__":
    data = np.array([[1,1,1], [3,3,4]])
    target = np.array([1,2,3,4])
    print(diff_data(data))
    print(diff_target(target))