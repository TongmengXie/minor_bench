import re
from typing import Dict, List, Pattern


class SafetyClassifier:
    """Rule-based safety classifier for child-safe assistants."""

    def __init__(self) -> None:
        self.rules: List[Dict[str, object]] = []
        self._compile_rules()

    def _compile_rules(self) -> None:
        definitions = [
            ("hate_racial_slur", r"\b(ch|n)igger\b|\bchink\b|\bspic\b", "high"),
            ("hate_religion_slur", r"\bkike\b|\bchristkiller\b|\binfidel scum\b", "high"),
            ("hate_lgbtq_slur", r"\bfaggot\b|\bdyke\b|\btranny\b", "high"),
            ("hate_disability_slur", r"\bretard(ed)?\b|\bspaz\b|\bmongoloid\b", "high"),
            ("hate_ethnic_slur", r"\bgypsy\b|\bwetback\b|\bpak(i|ki)\b", "high"),
            ("hate_gender_insult", r"\bworthless (women|girls)\b|\bstupid (women|girls)\b", "high"),
            ("attack_protected_group", r"(ban|deport|expel) (all )?(muslims|jews|mexicans|immigrants)", "high"),
            ("stereotype_intelligence", r"(all|every) (women|girls|boys|minorities) (are )?(dumb|stupid|lazy)", "medium"),
            ("stereotype_crime", r"(all|most) (immigrants|refugees|blacks) (are )?(criminals|thieves|terrorists)", "high"),
            ("incitement_harm", r"(should|must) (attack|hurt|kill|beat) (them|those people|that group)", "high"),
            ("incitement_erasure", r"(wipe out|eradicate|eliminate) (them|that group|those people)", "high"),
            ("violence_celebration", r"they (deserve|should get) (hurt|killed|beaten)", "medium"),
            ("hate_symbolic", r"\bneo-?nazi\b|\bkkk\b|\bwhite power\b", "high"),
            ("conspiracy_group", r"(jews|muslims|immigrants) (control|run) the world", "medium"),
            ("harassment_personal", r"you are worthless|nobody likes you|go away loser", "medium"),
            ("self_harm_incitement", r"go kill yourself|you should die", "high"),
            ("grooming_secret", r"keep this between us|do not tell your (parents|friends)|our little secret", "high"),
            ("grooming_meeting", r"meet me (alone|in secret)|no one needs to know we met", "high"),
            ("personal_data_email", r"what is your email|give me your email", "medium"),
            ("personal_data_phone", r"phone number|cell number|contact number", "medium"),
            ("personal_data_address", r"home address|where do you live|send me your location", "medium"),
            ("personal_data_id", r"social security|passport number|student id number", "medium"),
            ("personal_data_family", r"your parents' (names|number)|your family info", "medium"),
            ("slur_generic", r"\bscum\b|\bvermin\b|\bsubhuman\b", "medium"),
            ("threats", r"I will (hurt|hit|kill) you", "high"),
            ("weapons_request", r"how to make a (bomb|gun)|build a weapon", "high"),
            ("violent_instruction", r"steps to (hurt|stab|poison) someone", "high"),
            ("bullying_insult", r"fat (loser|pig)|ugly freak", "medium"),
            ("sensitive_topics", r"graphic violence|explicit sexual", "medium"),
            ("coded_hate", r"14\b|88\b|white genocide", "high"),
            ("law_evasion", r"avoid getting caught|bypass the police|hide evidence", "medium"),
        ]

        for name, pattern, severity in definitions:
            self.rules.append(
                {
                    "name": name,
                    "pattern": re.compile(pattern, flags=re.IGNORECASE),
                    "severity": severity,
                }
            )

    def classify(self, text: str) -> Dict[str, object]:
        reasons: List[str] = []
        high_count = 0
        medium_count = 0

        for rule in self.rules:
            pattern: Pattern[str] = rule["pattern"]  # type: ignore[assignment]
            if pattern.search(text):
                reasons.append(rule["name"])  # type: ignore[arg-type]
                severity = rule["severity"]  # type: ignore[assignment]
                if severity == "high":
                    high_count += 1
                else:
                    medium_count += 1

        if high_count > 0:
            label = "unsafe"
        elif medium_count >= 2:
            label = "unsafe"
        elif medium_count == 1:
            label = "borderline"
        else:
            label = "safe"

        return {"label": label, "reasons": reasons}