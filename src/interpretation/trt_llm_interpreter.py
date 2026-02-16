from __future__ import annotations

from typing import Iterable, Mapping, Any, Optional, List

from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner


class TrtLlmInterpreter:
    def __init__(
        self,
        engine_dir: str,
        tokenizer_dir: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.runner = ModelRunner.from_dir(engine_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.system_prompt = system_prompt or (
            "Tu es un expert veterinaire. Tu produis un rapport concis et factuel."
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def build_prompt(self, detections: Iterable[Mapping[str, Any]]) -> str:
        lines: List[str] = []
        for det in detections:
            species = det.get("species", "inconnu")
            temp = det.get("temp", "N/A")
            conf = det.get("conf", det.get("confidence", "N/A"))
            zone = det.get("zone", "N/A")
            lines.append(f"- {species}: temp {temp}C, conf {conf}, zone {zone}")

        user = "Detections:\n" + "\n".join(lines)
        user += "\n\nDemande: 1) anomalies 2) actions"
        return f"System: {self.system_prompt}\nUser: {user}\nAssistant:"

    def interpret(self, detections: Iterable[Mapping[str, Any]]) -> str:
        prompt = self.build_prompt(detections)
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        outputs = self.runner.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
