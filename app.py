from flask import Flask, render_template, Response, redirect, url_for, request, jsonify
import cv2
import time
import copy
from slr.main import process_frame  # Importer directement la fonction process_frame

app = Flask(__name__, template_folder="templates", static_folder="static")

# Variables globales
cap = None
generated_phrase = ""
last_sign_time = time.time()
letter_detect_time = time.time()
delay_between_letters = 2  # Délai entre chaque lettre (en secondes)
delay_threshold_for_space = 3  # 3 secondes de pause avant d'ajouter un espace
capture_flag = False  # Flag pour savoir si l'utilisateur a cliqué sur "Start Predict"

@app.route("/")
def index():
    """Page principale avec la vidéo et le résultat"""
    global generated_phrase
    return render_template("index.html", output=generated_phrase)

@app.route("/video")
def video():
    """Stream vidéo avec détection des signes"""
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)  # Initialisation de la caméra
        if not cap.isOpened():
            return "Error: Camera not available", 500

    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture", methods=["POST"])
def capture():
    """Capture une image depuis la webcam et active la reconnaissance"""
    global capture_flag
    capture_flag = True
    return redirect(url_for("index"))

@app.route("/reset", methods=["GET"])
def reset():
    """Réinitialiser la phrase générée"""
    global generated_phrase
    generated_phrase = ""
    return redirect(url_for("index"))

@app.route("/del_last", methods=["GET"])
def del_last():
    """Supprimer la dernière lettre"""
    global generated_phrase
    if generated_phrase:
        generated_phrase = generated_phrase[:-1]
    return redirect(url_for("index"))

@app.route("/get_prediction")
def get_prediction():
    """Renvoie la prédiction actuelle sous forme de JSON"""
    return jsonify({'generated_phrase': generated_phrase})

def gen_frames():
    """Générer les images vidéo avec les lettres détectées"""
    global cap, generated_phrase, last_sign_time, letter_detect_time, capture_flag

    if cap is None:
        cap = cv2.VideoCapture(0)  # Initialisation de la caméra
        if not cap.isOpened():
            raise RuntimeError("Camera not available")

    while True:
        success, image = cap.read()
        if not success:
            break

        # Traitement des images
        image = cv2.resize(image, (640, 480))
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Si l'utilisateur a cliqué sur le bouton, on commence à analyser les signes
        if capture_flag:
            detected_letter, debug_image, left_hand_detected = process_frame(debug_image)  # Ajouter un paramètre pour détecter la main gauche

            current_time = time.time()
            if detected_letter:
                if current_time - letter_detect_time > delay_between_letters:
                    # Ajouter un espace si la pause entre les lettres est plus longue que 3 secondes
                    if current_time - last_sign_time > delay_threshold_for_space:
                        generated_phrase += " "
                    generated_phrase += detected_letter
                    last_sign_time = current_time
                    letter_detect_time = current_time
            
            # Vérification si la main gauche est détectée, ajoute un espace
            if left_hand_detected:  # Si la main gauche est détectée, ajouter un espace
                if current_time - last_sign_time > delay_threshold_for_space:  # Éviter d'ajouter trop d'espaces
                    generated_phrase += " "
                    last_sign_time = current_time

        # Encoder l'image pour le flux vidéo
        _, buffer = cv2.imencode(".jpg", debug_image)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        if cap is not None:
            cap.release()
