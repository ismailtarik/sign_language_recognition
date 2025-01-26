import time  # Pour ajouter un délai
import copy
import csv
import os
import datetime

import pyautogui
import cv2 as cv
import mediapipe as mp
from dotenv import load_dotenv

from slr.model.classifier import KeyPointClassifier

from slr.utils.args import get_args
from slr.utils.cvfpscalc import CvFpsCalc
from slr.utils.landmarks import draw_landmarks

from slr.utils.draw_debug import get_result_image
from slr.utils.draw_debug import get_fps_log_image
from slr.utils.draw_debug import draw_bounding_rect
from slr.utils.draw_debug import draw_hand_label
from slr.utils.draw_debug import show_fps_log
from slr.utils.draw_debug import show_result

from slr.utils.pre_process import calc_bounding_rect
from slr.utils.pre_process import calc_landmark_list
from slr.utils.pre_process import pre_process_landmark

from slr.utils.logging import log_keypoints
from slr.utils.logging import get_dict_form_list
from slr.utils.logging import get_mode


import cv2 as cv
import mediapipe as mp
import copy
import csv
from slr.utils.pre_process import calc_bounding_rect, calc_landmark_list, pre_process_landmark
from slr.utils.draw_debug import draw_bounding_rect, draw_hand_label
from slr.utils.landmarks import draw_landmarks
from slr.model.classifier import KeyPointClassifier

def process_frame(image):
    """
    Processus de détection des mains et des signes sur une seule image.
    Retourne la lettre détectée et l'image annotée pour débogage.
    """
    # Initialisation des modules et modèles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    keypoint_classifier = KeyPointClassifier()

    keypoint_labels_file = "slr/model/label.csv"
    with open(keypoint_labels_file, encoding="utf-8-sig") as f:
        key_points = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in key_points]

    # Prétraitement de l'image
    debug_image = copy.deepcopy(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Processus de détection des mains
    results = hands.process(image)
    image.flags.writeable = True

    detected_letter = ""
    left_hand_detected = False  # Variable pour savoir si la main gauche est détectée

    # Si une main est détectée
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Vérifie si c'est la main droite
            if handedness.classification[0].label == "Right":
                # Calcul des points de repère
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Classifier le signe de la main
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                if hand_sign_id != 25:  # Ignorer si aucun signe détecté
                    detected_letter = keypoint_classifier_labels[hand_sign_id]

                # Dessiner les annotations pour débogage
                debug_image = draw_bounding_rect(debug_image, True, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_hand_label(debug_image, brect, handedness)

            # Vérifie si c'est la main gauche
            if handedness.classification[0].label == "Left":
                # Si la main gauche est détectée, définir left_hand_detected à True
                left_hand_detected = True

                # Dessiner les annotations pour la main gauche (optionnel)
                brect_left = calc_bounding_rect(debug_image, hand_landmarks)
                debug_image = draw_bounding_rect(debug_image, False, brect_left)  # Dessiner la main gauche en débogage

    return detected_letter, debug_image, left_hand_detected

def main():
    load_dotenv()
    args = get_args()

    keypoint_file = "slr/model/keypoint.csv"
    counter_obj = get_dict_form_list(keypoint_file)

    CAP_DEVICE = args.device
    CAP_WIDTH = args.width
    CAP_HEIGHT = args.height

    USE_STATIC_IMAGE_MODE = True
    MAX_NUM_HANDS = args.max_num_hands
    MIN_DETECTION_CONFIDENCE = args.min_detection_confidence
    MIN_TRACKING_CONFIDENCE = args.min_tracking_confidence

    USE_BRECT = args.use_brect
    MODE = args.mode
    DEBUG = int(os.environ.get("DEBUG", "0")) == 1
    CAP_DEVICE = 0

    print("INFO: System initialization Successful")
    print("INFO: Opening Camera")

    cap = cv.VideoCapture(CAP_DEVICE)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    
    background_image = cv.imread("resources/background.png")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=USE_STATIC_IMAGE_MODE,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    keypoint_classifier = KeyPointClassifier()

    keypoint_labels_file = "slr/model/label.csv"
    with open(keypoint_labels_file, encoding="utf-8-sig") as f:
        key_points = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in key_points]

    cv_fps = CvFpsCalc(buffer_len=10)
    print("INFO: System is up & running")

    # Pour stocker la phrase générée
    generated_phrase = ""
    last_sign_time = time.time()  # Pour gérer les pauses entre les lettres
    letter_detect_time = time.time()  # Gère le délai entre chaque lettre détectée
    delay_between_letters = 2  # Délai entre chaque lettre (en secondes)
    delay_threshold_for_space = 3  # 3 secondes de pause avant d'ajouter un espace

    while True:
        fps = cv_fps.get()

        key = cv.waitKey(1)
        if key == 27:  # ESC key
            print("INFO: Exiting...")
            break
        elif key == 57:  # 9
            name = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
            myScreenshot = pyautogui.screenshot()
            myScreenshot.save(f'ss/{name}.png')

        success, image = cap.read()
        if not success:
            continue
        
        image = cv.resize(image, (CAP_WIDTH, CAP_HEIGHT))
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        result_image = get_result_image()
        fps_log_image = get_fps_log_image()

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if DEBUG:
            MODE = get_mode(key, MODE)
            fps_log_image = show_fps_log(fps_log_image, fps)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                use_brect = True
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                if MODE == 0:
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    if hand_sign_id == 25:
                        hand_sign_text = ""
                    else:
                        hand_sign_text = keypoint_classifier_labels[hand_sign_id]

                    current_time = time.time()

                    # Vérifier si le délai de 2 secondes entre les lettres est écoulé
                    if current_time - letter_detect_time > delay_between_letters:
                        if hand_sign_text != "":
                            # Ajouter un espace si la pause entre les lettres est plus longue que 3 secondes
                            if current_time - last_sign_time > delay_threshold_for_space:
                                generated_phrase += " "
                            generated_phrase += hand_sign_text
                            last_sign_time = current_time  # Mettre à jour le temps de la dernière détection
                            letter_detect_time = current_time  # Mettre à jour le temps de détection de la lettre
                            print(f"Phrase générée jusqu'à présent : {generated_phrase}")

                    result_image = show_result(result_image, handedness, hand_sign_text)

                elif MODE == 1:
                    log_keypoints(key, pre_processed_landmark_list, counter_obj, data_limit=1000)

                debug_image = draw_bounding_rect(debug_image, use_brect, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_hand_label(debug_image, brect, handedness)

        background_image[170:170 + 480, 50:50 + 640] = debug_image
        background_image[240:240 + 127, 731:731 + 299] = result_image
        background_image[678:678 + 30, 118:118 + 640] = fps_log_image

        cv.imshow("Sign Language Recognition", background_image)

    cap.release()
    cv.destroyAllWindows()

    print("INFO: Bye")


if __name__ == "__main__":
    main()
