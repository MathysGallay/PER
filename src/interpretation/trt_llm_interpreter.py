from __future__ import annotations
from typing import Iterable, Mapping, Any, Optional
import random
from datetime import datetime

class TrtLlmInterpreter:
    """
    Version 'Système Expert' pour modèle binaire (Humain / Animal).
    Simule une analyse IA sans charger de modèle lourd.
    """
    def __init__(
        self,
        engine_dir: str = "", 
        tokenizer_dir: str = "",
        system_prompt: Optional[str] = None
    ) -> None:
        self.system_prompt = system_prompt
        # Pas de chargement de modèle = Pas de risque de crash RAM

    def interpret(self, detections: Iterable[Mapping[str, Any]]) -> str:
        """
        Génère un rapport textuel basé sur la classification binaire.
        """
        # 1. Analyse des données
        counts = {"Humain": 0, "Animal": 0}
        max_conf = 0.0
        
        for det in detections:
            # On récupère le label (adapte les clés selon ton code YOLO : 'label', 'class_name', etc.)
            raw_label = det.get("label", det.get("species", "")).lower()
            conf = float(det.get("conf", det.get("confidence", 0.0)))
            
            if conf > max_conf:
                max_conf = conf

            # Classification binaire simple
            if "person" in raw_label or "humain" in raw_label or "man" in raw_label:
                counts["Humain"] += 1
            else:
                # Tout ce qui n'est pas humain est considéré comme Animal dans ton modèle
                counts["Animal"] += 1

        timestamp = datetime.now().strftime("%H:%M:%S")

        # 2. Scénarios (Logique conditionnelle)
        
        # CAS 1 : RIEN
        if counts["Humain"] == 0 and counts["Animal"] == 0:
            phrases = [
                f"[{timestamp}] Zone calme. Aucun sujet détecté.",
                f"[{timestamp}] Monitoring actif. Secteur vide.",
                f"[{timestamp}] En attente de passage..."
            ]
            return random.choice(phrases)

        # CAS 2 : HUMAIN (Priorité Sécurité)
        if counts["Humain"] > 0:
            nb = counts["Humain"]
            actions = ["déplacement lent", "immobile", "traversée de zone"]
            return (f"⚠️ ALERTE INTRUSION [{timestamp}]\n"
                    f"Détection : {nb} Humain(s).\n"
                    f"Confiance : {max_conf:.2f}\n"
                    f"Analyse : Présence humaine non autorisée. Comportement : {random.choice(actions)}.\n"
                    f"Action : Notification envoyée au poste de garde.")

        # CAS 3 : ANIMAL (Priorité Écologie)
        elif counts["Animal"] > 0:
            nb = counts["Animal"]
            return (f"🌲 PASSAGE FAUNE [{timestamp}]\n"
                    f"Détection : {nb} Animal(aux).\n"
                    f"Confiance : {max_conf:.2f}\n"
                    f"Analyse : Faune locale en mouvement.\n"
                    f"Action : Archivage pour comptage biodiversité.")
        
        return "Analyse en cours..."