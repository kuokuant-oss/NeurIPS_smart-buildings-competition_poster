"""Dashboard v1 index (2026-10-30): data/runs/dashboard/index.json for the viewer (index.html), rebuilt from the
records and building files that exist. Holds the milestone list (order, Chinese label, public score), the scenarios,
which record files exist, each record's episode summary (for the improvement chart) and the reward-part check.

python3 -B pipeline/launch/launch.py pipeline/setup/dash_index.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/runs/dashboard"

# Milestones in submission order (public leaderboard score, lower is better).
MILESTONES = [
    {"key": "rule_v4", "source": "G36PolicyC3LaaOccrowDampfloor (packages/smart_building/policies/g36.py)", "label": "規則第 4 版（較早的規則）", "short": "規則第 4 版", "public": 0.01647},
    {"key": "rule", "source": "G36PolicyB2R2 (packages/smart_building/policies/g36.py)", "label": "原規則", "short": "原規則", "public": 0.01441},
    {"key": "fixed-u100", "source": "data/runs/b2rl/offset-check/model-fixed-u100.json", "label": "固定參數版", "short": "固定參數", "public": 0.01404},
    {"key": "s5_sched_topt3600_hot_safe", "source": "data/runs/b2rl/search/models/model-s5_sched_topt3600_hot_safe.json", "label": "依外氣排程", "short": "依外氣排程", "public": 0.01370},
    {"key": "f2d_a12_g05", "source": "data/runs/b2rl/search/models/model-f2d_a12_g05.json", "label": "成本式送風溫度＋保護 0.5 度", "short": "成本式送風溫度", "public": 0.01330},
    {"key": "band_pi_empty", "source": "data/runs/b2rl/search/models/model-band_pi_empty.json", "label": "三項全加：依外氣保護、區迴路、空日偵測", "short": "三項全加", "public": 0.01281},
    {"key": "night1_dcv07", "source": "data/runs/b2rl/search/models/model-night1_dcv07.json", "label": "整夜重調＋依人數最低風量（10/30 提交）", "short": "整夜重調＋依人數風量",
     "public": 0.01152},
    # 2026-11-10 (user: "加進樓層圖"): the per-room tree era, named by content; public scores from
    # docs/experiments/leaderboard-log.md (None = not scored yet; fill in when the score arrives)
    {"key": "gt_u1b", "source": "data/runs/b2rl/search/models/model-gt_u1b.json",
     "label": "每房間樹第一版：依房間面積、外牆與外氣，從 15 個設定選一個", "short": "每房間樹第一版", "public": 0.01108},
    {"key": "gt_r5pa2", "source": "data/runs/b2rl/search/models/model-gt_r5pa2.json",
     "label": "論文規則版：只用物理選項的規則（論文比較基準）", "short": "論文規則版", "public": 0.01064},
    {"key": "gt_wub", "source": "data/runs/b2rl/search/models/model-gt_wub.json",
     "label": "論文規則版＋整棟氣候門檻與房間格設定（規則擬合樹）", "short": "整棟門檻樹", "public": 0.01067},
    {"key": "gt_r3a1", "source": "data/runs/b2rl/search/models/model-gt_r3a1.json",
     "label": "第三輪樹：整棟門檻＋房間決策樹", "short": "第三輪樹（房間樹）", "public": 0.01069},
    {"key": "gt_r3d1", "source": "data/runs/b2rl/search/models/model-gt_r3d1.json",
     "label": "第三輪樹：整棟門檻＋房間線性模型（含鄰房圖幾何）", "short": "第三輪樹（鄰房圖）", "public": 0.01069},
    {"key": "gt_m1", "source": "data/runs/b2rl/search/models/model-gt_m1.json",
     "label": "合併版一：第三輪樹（鄰房圖）＋寒冷時只降不加熱房間的最小風量＋大房間需要才加熱（上限 0.6）",
     "short": "合併版一", "public": None},
    {"key": "gt_m2", "source": "data/runs/b2rl/search/models/model-gt_m2.json",
     "label": "合併版二：合併版一＋沒人來的平日延後預冷＋嚴寒時經濟目標離舒適邊界 0.6 度", "short": "合併版二", "public": None},
    {"key": "gt_b2", "source": "data/runs/b2rl/search/models/model-gt_b2.json",
     "label": "規則＋四手段（不含樹）：論文規則版＋合併版二的四個規則手段", "short": "規則＋四手段", "public": None},
]
# Stage 2B zone-geometry experiments (2026-10-30, rounds 24-26; night1_dcv07 plus one G36Config switch each; none was
# submitted and all were worse than night1_dcv07 on TRAIN 100), shown after the milestones for the user's own analysis.
EXPERIMENTS = [
    {"key": "geo_h05", "source": "data/runs/b2rl/search/models/model-geo_h05.json",
     "label": "實驗：外牆區暖房上限調高（強度 0.5）", "short": "實驗：暖房上限調高", "public": None},
    {"key": "geo_r05", "source": "data/runs/b2rl/search/models/model-geo_r05.json",
     "label": "實驗：空調箱內重新分配風量（強度 0.5）", "short": "實驗：重新分配風量", "public": None},
    {"key": "geo_hm05", "source": "data/runs/b2rl/search/models/model-geo_hm05.json",
     "label": "實驗：外牆區暖房上限調低（強度 0.5）", "short": "實驗：暖房上限調低", "public": None},
    {"key": "geo_q75_m025", "source": "data/runs/b2rl/search/models/model-geo_q75_m025.json",
     "label": "實驗：溫和天氣保護只看大區（前 25%）", "short": "實驗：保護只看大區", "public": None},
    # 2026-10-31 per-room heating boost (zone_heat_boost_max 2.0): train_fairbanks1 +23% vs night1_dcv07; recorded on
    # two cold cases only, to show why (the boxes' coils are at their hot-water limit; more air, no more heat)
    {"key": "hb20", "source": "data/runs/b2rl/search/models/model-hb20.json",
     "label": "實驗：缺熱房間加大風量（最多 2 倍，只錄兩棟嚴寒）", "short": "實驗：缺熱房間加風量", "public": None},
]
# The strongest unsubmitted candidate (user 2026-10-31: "主要是跟我們目前最強的候選比"): the condition-based rule
# cond40h on night1_dcv07 (block decisions from forests, daily from trees; fast simulator against night1_dcv07: held-out
# TRAIN 60 -1.41%, INNER 29 -1.19%, S4 first 30 -0.05%), recorded through pipeline/setup/cond_policy.py.
EXPERIMENTS.insert(0, {"key": "cond_cond40h", "source": "cond=model-night1_dcv07.json=rules/cond40h.json",
                       "label": "候選：條件式規則（依天氣與室溫切換選項，尚未提交）", "short": "候選：條件式規則",
                       "public": None})
# What each case shows (from the dashboard records and the stage 2 zone analysis), shown under the controls.
GUIDE = {
    "train/building_11818/fairbanks_1": "看點：上班時段外氣平均約 −22 °C。10/30 的整夜重調版仍有不少區整月偏冷，最冷的是外牆長的角落區（4、16、47、52、62 號）；外牆暴露與各區舒適成本的相關約 0.70。",
    "train/building_15790/fairbanks_1": "看點：同樣嚴寒，但這棟的冷區和外牆暴露幾乎無關（相關約 0），反而是面積小的區較冷（與面積相關 −0.47）。",
    "val/building_31441/fairbanks_1": "看點：最極端的嚴寒。5 個區（外牆 30–88 公尺）整個上班時段都冷；從原規則到10/30 的整夜重調版，其他區改善約 80%，外牆最多的四分之一區反而差 7%。",
    "val/building_31476/fairbanks_1": "看點：嚴寒。最差的 10% 區佔了 80% 的舒適成本，外牆最多的四分之一區佔 84%。",
    "train/building_11534/singapore_5": "看點：138 區的大建築、濕熱。從原規則到10/30 的整夜重調版，外牆最多的四分之一區反而變差（−17%），其他區改善 62%。",
    "val/building_32725/singapore_5": "看點：成本九成是冷卻用電，各版本幾乎沒差，接近物理下限；最熱的是面積最大的區（943 平方公尺）。",
    "val/building_39445/phoenix_5": "看點：乾熱。最差的區都是大區（650–930 平方公尺，約中位數的兩倍），風量開到變風量箱上限仍然太熱。",
    "val/building_31750/atlanta_7": "看點：排行榜預估場景中，10/30 版本退步的那棟（成本式送風溫度版 0.01093 → 整夜重調版 0.01209）；問題集中在大區太熱（與面積相關 0.74）。",
    "train/building_12567/atlanta_5": "看點：從原規則到10/30 的整夜重調版，改善幾乎都在外牆最多的區（約 92%），其他區幾乎沒變。",
    "train/building_16909/denver_5": "看點：早期版本一個區接近溫度上限，就把整組都吹冷；10/30 的整夜重調版成本約從 0.034 降到 0.008。",
    "val/building_37294/denver_5": "看點：改善最多的例子之一（約 0.025 → 0.006）；外牆暴露多、面積小的區問題較多。",
    "inner/building_20369/denver_1": "看點：乾冷。外牆暴露與舒適成本相關 0.54、與面積相關 −0.50；外牆區與其他區都改善約 65–73%。",
}

CITY = {"fairbanks": "費爾班克斯（阿拉斯加，嚴寒）", "denver": "丹佛（乾冷）", "singapore": "新加坡（濕熱）",
        "atlanta": "亞特蘭大（濕熱夏季）", "phoenix": "鳳凰城（乾熱）"}
POOL = {"train": "訓練集", "val": "驗證集", "inner": "內部驗證集"}
REASON = {
    "train/building_11818/fairbanks_1": "嚴寒、難保溫",
    "train/building_15790/fairbanks_1": "嚴寒、難保溫",
    "val/building_31441/fairbanks_1": "嚴寒、難保溫",
    "val/building_31476/fairbanks_1": "嚴寒、難保溫",
    "train/building_11534/singapore_5": "濕熱、以冷房為主",
    "val/building_32725/singapore_5": "濕熱、以冷房為主",
    "val/building_39445/phoenix_5": "乾熱、以冷房為主",
    "val/building_31750/atlanta_7": "10/30 版本在此退步",
    "train/building_12567/atlanta_5": "溫和",
    "train/building_16909/denver_5": "溫和",
    "val/building_37294/denver_5": "溫和",
    "inner/building_20369/denver_1": "溫和",
}


def scenario_label(spec):
    pool, building, weather = spec.split("/")
    city, _, n = weather.rpartition("_")
    return (f"建築 {building.split('_')[-1]} ・ {CITY.get(city, city)} 天氣 {n} ・ {POOL.get(pool, pool)}"
            + (f" ・ {REASON[spec]}" if spec in REASON else ""))


def tree_note(recs):
    """2026-11-10: one data line per case for the tree-era versions (paper rule -> per-room tree -> merged two)."""
    a, t, b = recs.get("gt_r5pa2"), recs.get("gt_r3d1"), recs.get("gt_m2")
    if not (a and b):
        return ""
    sa, sb = a["summary"], b["summary"]
    pct = lambda s: f"{100 * (s['cost'] - sa['cost']) / sa['cost']:+.1f}%".replace("-", "−")
    k = lambda v: f"{1000 * v:+.3f}".replace("-", "−")
    mid = f"第三輪樹（鄰房圖）{t['summary']['cost']:.4f}（{pct(t['summary'])}）→ " if t else ""
    return (f" 11/10 新版：論文規則版 {sa['cost']:.4f} → {mid}合併版二 {sb['cost']:.4f}（{pct(sb)}；舒適 "
            f"{k(sb['comfort'] - sa['comfort'])}、能源＋碳 {k(sb['energy'] + sb['carbon'] - sa['energy'] - sa['carbon'])}，"
            f"每步成本 × 1000）；超出 1 小時以上的區 {sa['zones_out']} → {sb['zones_out']}。")


def build():
    records = {}
    for path in sorted((OUT / "records").glob("*.json")):
        if path.name.startswith(("smoke__", ".")):
            continue
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        zs = r["zone_summary"]
        summary = {k: r["summary"][k] for k in ("cost", "comfort", "energy", "carbon", "parts", "steps",
                                                "occupied_steps", "check_max_abs")}
        summary.update(dh_cold=round(sum(zs["dh_cold"]), 2), dh_hot=round(sum(zs["dh_hot"]), 2),
                       occ_hours_out=round(sum(zs["occ_hours_out"]), 1),
                       zones_out=sum(1 for v in zs["occ_hours_out"] if v >= 1.0), zones=len(r["zones"]),
                       temp_check_k=r["summary"]["check"]["obs_vs_reward_zone_temp_max_k"])
        records.setdefault(r["spec"], {})[r["policy"]] = {"file": f"records/{path.name}", "summary": summary}
    buildings = {p.stem: f"buildings/{p.name}" for p in sorted((OUT / "buildings").glob("building_*.json"))}
    scenarios = []
    for spec in sorted(records, key=lambda s: (list(REASON).index(s) if s in REASON else 99, s)):
        building = spec.split("/")[1]
        scenarios.append({"spec": spec, "building": building, "label": scenario_label(spec),
                          "guide": GUIDE.get(spec, "") + tree_note(records[spec]), "building_file": buildings.get(building),
                          "records": records[spec]})
    known = MILESTONES + EXPERIMENTS
    keys = {m["key"] for m in known}
    extra = sorted({p for s in records.values() for p in s} - keys)
    milestones = known + [{"key": k, "label": k, "short": k, "public": None} for k in extra]
    used = {p for s in records.values() for p in s}
    index = {"milestones": [m for m in milestones if m["key"] in used], "scenarios": scenarios,
             "check_max_abs": max((r["summary"]["check_max_abs"] for s in records.values() for r in s.values()),
                                  default=None),
             "temp_check_max_k": max((r["summary"]["temp_check_k"] for s in records.values() for r in s.values()),
                                     default=None)}
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return index


if __name__ == "__main__":
    idx = build()
    print(len(idx["scenarios"]), "scenarios;", sum(len(s["records"]) for s in idx["scenarios"]), "records;",
          "max check", idx["check_max_abs"])
