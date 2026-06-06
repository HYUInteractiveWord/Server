RANK_THRESHOLDS = [
    (0, "Bronze"),
    (500, "Silver"),
    (1000, "Gold"),
    (1500, "Sapphire"),
    (2000, "Ruby"),
    (2500, "Emerald"),
    (3000, "Amethyst"),
    (3500, "Pearl"),
    (4000, "Obsidian"),
    (4500, "Diamond"),
]

RANK_WORD_SLOTS = {
    "Bronze": 20,
    "Silver": 40,
    "Gold": 60,
    "Sapphire": 80,
    "Ruby": 100,
    "Emerald": 110,
    "Amethyst": 120,
    "Pearl": 130,
    "Obsidian": 140,
    "Diamond": 150,
}

# 단어 레벨업 기준 점수 (1~5레벨)
LEVEL_THRESHOLDS = [0, 20, 50, 65, 85, 100]


def get_rank_for_xp(xp: int) -> str:
    rank = "Bronze"
    for threshold, name in RANK_THRESHOLDS:
        if xp >= threshold:
            rank = name
    return rank


def calculate_xp_gain(score: float, is_new_best: bool) -> int:
    base = 20
    if is_new_best:
        base += 10
    if score >= 90:
        base += 25
    elif score >= 70:
        base += 20
    elif score >= 50:
        base += 15
    return base


def update_word_level(current_level: int, new_score: float) -> int:
    for level in range(5, 0, -1):
        if new_score >= LEVEL_THRESHOLDS[level]:
            return max(current_level, level)
    return current_level
