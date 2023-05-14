import csv
import cv2 as cv
import json


def log_keypoints(key, landmark_list, data_limit=1000):
    """

    :param key: Keyboard key (latter)
    :param landmark_list: Preprocessed landmark list
    :param data_limit: How many row need for each sign
    :return: None
    """
    counter_file = "model/counter.json"
    counter_obj = {}
    csv_path = "model/keypoint.csv"
    index = -1

    with open(counter_file, "r") as cf:
        counter_obj = dict(json.load(cf))

    #: Escaping 'J/j'
    if key == 106 or key == 74:
        return

    if 65 <= key <= 89 or 97 <= key <= 121:  # A-Z / a-z

        #: Calculating index of letters
        if 65 <= key <= 90:  # Capital letters
            index = key - 65

            #: Subtracting index by 1 after 'J'
            if key > 74:   # J
                index -= 1

        elif 97 <= key <= 122:  # Small letters
            index = key - 97

            #: Subtracting index by 1 after 'j'
            if key > 106:   # j
                index -= 1

        #: Counting limit
        if str(index) in counter_obj.keys():
            counter_obj[str(index)] += 1
        else:
            counter_obj[str(index)] = 1

        if counter_obj[str(index)] > data_limit:  #: Limit of capturing image
            print(f"Dataset limit reached for {chr(key).upper()} [{counter_obj[str(index)]-1}/{data_limit}]")
            return

        #: -
        #: Writing dataset
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([index, *landmark_list])


        #: -
        #: Writing counter
        with open(counter_file, "w") as cf:
            counter_obj_writable = json.dumps(counter_obj, indent=4)
            cf.write(counter_obj_writable)

        print(f"{chr(key).upper()} => {counter_obj[str(index)]}/{data_limit}")

    return


def _get_alphabet_index(key):
    """

    :param key: Keyboard key (latter)
    :return: Index of alphabate
    """
    cap_ascii_list = [
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
        78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90
    ]
    sm_ascii_list = [
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122
    ]

    index = 0
    return index


def get_mode(key, _mode):
    """
    :param key: Pressed key
    :param mode: Mode of program
    :return: mode
    """
    mode = _mode
    if key == 48:
        mode = 0
    elif key == 49:
        mode = 1

    return mode
