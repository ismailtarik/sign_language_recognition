# gesture_phrases.py

GESTES_PHRASES = {
    "Right": "Vous avez levé la main droite.",
    "Left": "Vous avez levé la main gauche.",
    "Fist": "Vous avez fait un poing.",
    "Open": "Vous avez ouvert la main.",
    "Peace": "Vous avez fait un signe de paix.",
    "ThumbUp": "Vous avez fait un pouce en l'air."
}

def generate_phrase(handedness_label, hand_sign_text):
    """
    Génère une phrase basée sur le geste de la main détecté.
    :param handedness_label: Le côté de la main (gauche/droite)
    :param hand_sign_text: Le texte associé au geste
    :return: La phrase générée
    """
    # Exemple de base : phrase pour la main gauche/droite
    gesture_phrase = GESTES_PHRASES.get(handedness_label, "Geste inconnu.")

    # Si un texte de geste spécifique est fourni, l'ajouter à la phrase
    if hand_sign_text:
        gesture_phrase += f" Le geste détecté est : {hand_sign_text}."
    
    return gesture_phrase
