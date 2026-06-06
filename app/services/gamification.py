RANK_THRESHOLDS = [
    (0, "Bronze"),
    (500, "Silver"),
    (1500, "Gold"),
    (3000, "Sapphire"),
    (6000, "Ruby"),
    (10000, "Emerald"),
    (15000, "Amethyst"),
    (21000, "Pearl"),
    (28000, "Obsidian"),
    (36000, "Diamond"),
]

RANK_WORD_SLOTS = {
    "Bronze": 20,
    "Silver": 30,
    "Gold": 45,
    "Sapphire": 60,
    "Ruby": 80,
    "Emerald": 100,
    "Amethyst": 130,
    "Pearl": 160,
    "Obsidian": 200,
    "Diamond": 999,
}

# 단어 레벨업 기준 점수 (1~5레벨)
LEVEL_THRESHOLDS = [0, 60, 75, 85, 93, 98]


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
