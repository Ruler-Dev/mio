"""Fun / utility skills — random, dice, names, wordle helper, wikipedia."""

from __future__ import annotations

import json
import random
import re


# ============================================================
# Dice / coin / random picker
# ============================================================
def roll_dice(notation: str = "1d6") -> dict:
    """Roll dice in standard NdM notation (e.g. 2d6, 1d20, 4d6+3)."""
    m = re.match(r"^(\d+)d(\d+)(?:([+-])(\d+))?$", notation.strip().lower())
    if not m:
        return {"skill": "roll_dice", "error": f"invalid notation: {notation}"}
    n, sides = int(m.group(1)), int(m.group(2))
    mod_sign, mod_val = m.group(3), int(m.group(4) or 0)
    if n > 100 or sides > 1000:
        return {"skill": "roll_dice", "error": "too many / too large"}
    rolls = [random.randint(1, sides) for _ in range(n)]
    total = sum(rolls)
    if mod_sign == "+":
        total += mod_val
    elif mod_sign == "-":
        total -= mod_val
    return {
        "skill": "roll_dice",
        "notation": notation,
        "rolls": rolls,
        "modifier": (mod_sign or "") + str(mod_val) if mod_val else "",
        "total": total,
    }


def flip_coin(count: int = 1) -> dict:
    count = max(1, min(int(count), 1000))
    flips = [random.choice(["H", "T"]) for _ in range(count)]
    return {
        "skill": "flip_coin",
        "flips": flips,
        "heads": flips.count("H"),
        "tails": flips.count("T"),
    }


def pick_random(items: list, count: int = 1) -> dict:
    if not items:
        return {"skill": "pick_random", "error": "empty list"}
    count = max(1, min(int(count), len(items)))
    picked = random.sample(items, count)
    return {"skill": "pick_random", "picked": picked}


# ============================================================
# Name generator
# ============================================================
_ADJECTIVES = [
    "swift",
    "bright",
    "quiet",
    "bold",
    "clever",
    "merry",
    "silent",
    "wild",
    "sharp",
    "velvet",
    "iron",
    "crystal",
    "amber",
    "obsidian",
    "golden",
    "sapphire",
    "emerald",
    "radiant",
    "dusky",
    "hollow",
    "northern",
    "solar",
    "lunar",
    "arctic",
    "midnight",
    "dawn",
    "morning",
    "dusty",
    "wandering",
    "fable",
    "fabled",
    "hidden",
    "quiet",
    "pale",
    "copper",
    "gilded",
]
_NATURE = [
    "river",
    "mountain",
    "forest",
    "meadow",
    "ocean",
    "storm",
    "willow",
    "oak",
    "pine",
    "raven",
    "owl",
    "fox",
    "wolf",
    "bear",
    "heron",
    "lark",
    "tide",
    "ember",
    "cinder",
    "mist",
    "dusk",
    "feather",
    "petal",
    "grove",
    "crag",
    "harbor",
    "thicket",
    "vale",
    "fjord",
    "atoll",
    "cairn",
]
_TECH = [
    "bit",
    "byte",
    "fork",
    "link",
    "node",
    "proxy",
    "cache",
    "cluster",
    "shard",
    "stream",
    "relay",
    "beacon",
    "atlas",
    "pulse",
    "signal",
    "vertex",
    "orbit",
    "lambda",
    "kernel",
    "matrix",
]
_HUMAN_FIRST = [
    "Alex",
    "Jordan",
    "Sam",
    "Riley",
    "Morgan",
    "Cameron",
    "Quinn",
    "Avery",
    "Skyler",
    "Rowan",
    "Finley",
    "Emerson",
    "Hayden",
    "Dakota",
    "Sage",
    "River",
    "Kai",
    "Nova",
    "Phoenix",
    "Wren",
    "Ashe",
]
_HUMAN_LAST = [
    "Reyes",
    "Okafor",
    "Patel",
    "Chen",
    "Tanaka",
    "Silva",
    "Nakamura",
    "Hoffman",
    "Kouri",
    "Abiola",
    "Vasquez",
    "Dunn",
    "Larsen",
    "Kirby",
    "Ngata",
    "Ibarra",
    "Sokolov",
    "Zheng",
    "Kaur",
    "Moreau",
    "Park",
]


def generate_names(kind: str = "person", count: int = 5, theme: str = "") -> dict:
    """Generate names for people, companies, pets, products, etc."""
    count = max(1, min(int(count), 50))
    out: list[str] = []
    if kind == "person":
        out = [f"{random.choice(_HUMAN_FIRST)} {random.choice(_HUMAN_LAST)}" for _ in range(count)]
    elif kind == "company":
        pool1 = _NATURE + _TECH
        pool2 = ["Labs", "Works", "Studio", "Ventures", "Systems", "Forge", "Collective", "& Co", "Research", "Group"]
        out = [f"{random.choice(pool1).capitalize()} {random.choice(pool2)}" for _ in range(count)]
    elif kind == "product":
        out = [f"{random.choice(_ADJECTIVES).capitalize()}{random.choice(_NATURE).capitalize()}" for _ in range(count)]
    elif kind == "pet":
        pet_pool = [
            "Mochi",
            "Nori",
            "Biscuit",
            "Coco",
            "Peach",
            "Miso",
            "Ollie",
            "Clover",
            "Pepper",
            "Luna",
            "Bandit",
            "Hazel",
            "Remy",
            "Finn",
            "Ziggy",
            "Yuki",
            "Otto",
        ]
        out = [random.choice(pet_pool) for _ in range(count)]
    elif kind == "fantasy":
        syll1 = ["ar", "el", "val", "my", "tha", "jor", "kil", "mar", "sy", "bran", "rhen"]
        syll2 = ["dor", "dil", "ion", "ra", "wyn", "lith", "mir", "thas", "lon"]
        syll3 = ["", "iel", "as", "us", "or", "ic", "eth", "yn"]
        out = [random.choice(syll1).capitalize() + random.choice(syll2) + random.choice(syll3) for _ in range(count)]
    else:
        return {"skill": "generate_names", "error": f"unknown kind: {kind}"}
    # Bias toward theme keyword if provided
    if theme:
        out = [n for n in out if theme.lower() in n.lower()] + out
        out = out[:count]
    return {"skill": "generate_names", "kind": kind, "theme": theme, "names": out}


# ============================================================
# Wordle helper
# ============================================================
_WORDLE_WORDS_CACHE: list[str] = []


def _wordle_words() -> list[str]:
    # Small seed list for offline use; covers common 5-letter answers.
    global _WORDLE_WORDS_CACHE
    if _WORDLE_WORDS_CACHE:
        return _WORDLE_WORDS_CACHE
    seed = (
        "about above abuse actor acute admit adopt adult after again agent "
        "agree ahead alarm album alert alike alive allow alone along alter "
        "among anger angle angry apart apple apply arena argue arise array "
        "aside asset avoid awake award aware badly baker bases basic basis "
        "beach began begin begun being below bench billy birth black blame "
        "blind block blood board boost booth bound brain brand brass brave "
        "bread break breed brief bring broad broke brown build built buyer "
        "cable calif carry catch cause chain chair chart chase cheap check "
        "chest chief child china chose civil claim class clean clear click "
        "clock close coach coast could count court cover craft crash cream "
        "crime cross crowd crown curve cycle daily dance dated dealt death "
        "debut delay depth doing doubt dozen draft drama drawn dream dress "
        "drill drink drive drove dying eager early earth eight elite empty "
        "enemy enjoy enter entry equal error event every exact exist extra "
        "faith false fault fiber field fifth fifty fight final first fixed "
        "flash fleet floor fluid focus force forth forty forum found frame "
        "frank fraud fresh front fruit fully funny giant given glass globe "
        "going grace grade grand grant grass great green gross group grown "
        "guard guess guest guide happy harry heart heavy hence henry horse "
        "hotel house human ideal image index inner input issue japan jimmy "
        "joint jones judge known label large laser later laugh layer learn "
        "lease least leave legal level lewis light limit links lives local "
        "logic loose lower lucky lunch lying magic major maker march maria "
        "match maybe mayor meant media metal might minor minus mixed model "
        "money month moral motor mount mouse mouth movie needs never newly "
        "night noise north noted novel nurse occur ocean offer often order "
        "other ought paint panel paper party peace peter phase phone photo "
        "piece pilot pitch place plain plane plant plate point pound power "
        "press price pride prime print prior prize proof proud prove queen "
        "quick quiet quite radio raise range rapid ratio reach ready refer "
        "right rival river rough round route royal rural scale scene scope "
        "score sense serve seven shall shape share sharp sheet shelf shell "
        "shift shirt shock shoot short shown sight since sixth sixty sized "
        "skill sleep slide small smart smile smith smoke solid solve sorry "
        "sound south space spare speak speed spend spent split spoke sport "
        "staff stage stake stand start state steam steel steep steer stick "
        "still stock stone stood store storm story strip stuck study stuff "
        "style sugar suite super sweet table taken taste taxes teach teeth "
        "terry texas thank theft their theme there these thick thing think "
        "third those three threw throw tight times tired title today topic "
        "total touch tough tower track trade train treat trend trial tried "
        "tries truck truly trust truth twice under union until upper upset "
        "urban usage usual valid value video virus visit vital voice waste "
        "watch water wheel where which while white whole whose woman women "
        "world worry worse worst worth would wound write wrong wrote yield "
        "young youth"
    ).split()
    _WORDLE_WORDS_CACHE = [w for w in seed if len(w) == 5]
    return _WORDLE_WORDS_CACHE


def wordle_helper(
    green: str = "",
    yellow: str = "",
    grey: str = "",
    known_positions: dict | None = None,
    wrong_positions: dict | None = None,
) -> dict:
    """Suggest Wordle candidates.
    - `green`: letters known to be at specific positions, e.g. 'S---E'
      (use '-' for unknown slots; length must be 5).
    - `yellow`: letters that ARE in the word but position unknown.
    - `grey`: letters confirmed NOT in the word.
    - `wrong_positions`: {letter: [positions]} where that letter shouldn't be.
    """
    words = _wordle_words()
    green = (green or "-----").lower().ljust(5, "-")[:5]
    yellow_set = set((yellow or "").lower())
    grey_set = set((grey or "").lower()) - yellow_set - set(green) - {"-"}
    candidates = []
    for w in words:
        wl = w.lower()
        if len(wl) != 5:
            continue
        ok = True
        for i, g in enumerate(green):
            if g != "-" and wl[i] != g:
                ok = False
                break
        if not ok:
            continue
        if not all(y in wl for y in yellow_set):
            continue
        if any(gr in wl for gr in grey_set):
            continue
        if wrong_positions:
            for letter, positions in wrong_positions.items():
                for p in positions:
                    if 0 <= p < 5 and wl[p] == letter.lower():
                        ok = False
                        break
                if not ok:
                    break
        if not ok:
            continue
        candidates.append(wl)
    return {
        "skill": "wordle_helper",
        "count": len(candidates),
        "candidates": candidates[:30],
    }


# ============================================================
# Wikipedia summary
# ============================================================
def wiki_summary(topic: str, lang: str = "en") -> dict:
    """Fetch a Wikipedia summary for a topic (REST API, no key)."""
    import urllib.request as _urlreq
    import urllib.parse as _urlparse

    if not topic:
        return {"skill": "wiki_summary", "error": "topic required"}
    t = _urlparse.quote(topic.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{t}"
    try:
        req = _urlreq.Request(url, headers={"User-Agent": "mio/1.0"})
        d = json.loads(_urlreq.urlopen(req, timeout=10).read())
    except Exception as e:
        return {"skill": "wiki_summary", "error": f"fetch failed: {e}"}
    return {
        "skill": "wiki_summary",
        "title": d.get("title"),
        "extract": d.get("extract"),
        "url": (d.get("content_urls") or {}).get("desktop", {}).get("page"),
        "image": (d.get("thumbnail") or {}).get("source"),
        "description": d.get("description"),
    }
