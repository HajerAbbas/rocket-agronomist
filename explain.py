"""
explain.py - turns the deterministic measurement into (a) step events for the
live timeline and (b) plain-language "measurement reasoning". No AI here, so
the app stays useful even when the AI agent is offline.
"""

from __future__ import annotations

from typing import List, Optional

T = {
    "en": {
        "step_quality": "Checking image quality",
        "step_detect": "Detecting the leaf",
        "step_measure": "Measuring leaf colour",
        "step_estimate": "Estimating nitrogen status",
        "step_ref": "Saving reference plant",
        "sharp": "sharpness", "bright": "brightness", "blurry": "blurry", "dark": "too dark",
        "covers": "Leaf covers {p}% of the frame ({n} region{s})",
        "no_leaf": "No leaf found ({p}% green pixels)",
        "pixels": "{v}% of leaf pixels usable, glare {g}%",
        "skipped": "Skipped",
        "mode_cal": "Calibrated model: {v}% N",
        "mode_ref": "Compared with reference: SI {si} → {st}",
        "mode_idx": "Greenness measured (no calibration yet)",
        "ref_saved": "Reference plant saved (greenness {d})",
        "ref_failed": "No leaf found in the reference photo",
        # reasoning
        "r_quality_ok": "The photo is sharp and bright enough to measure.",
        "r_blurry": "The photo is blurry, so colour cannot be measured reliably.",
        "r_dark": "The photo is too dark; colour readings would be wrong.",
        "r_leaf": "A leaf was found covering {p}% of the frame.",
        "r_small": "The leaf is small in the frame, so fewer pixels were measured.",
        "r_noleaf": "Less than 5% of the frame is leaf-green, so no leaf was detected.",
        "r_glare": "{g}% of the leaf had flash glare and was excluded.",
        "r_dgci": "Leaf greenness (DGCI) is {d}: {word}.",
        "pale": "pale", "medium": "medium green", "dark_green": "dark green",
        "r_brown": "{b}% of the leaf looks brown, which may mean disease or old tissue rather than nitrogen.",
        "r_yellow": "{y}% of the leaf looks yellowish.",
        "r_ref": "Compared with your reference plant the leaf is at {pct}% of its greenness (sufficiency index {si}).",
        "r_ref_rule": "Rule: ≥ 95% sufficient, 90–95% borderline, < 90% deficient.",
        "r_cal": "The calibrated model estimates {v}% leaf nitrogen (range {lo}–{hi}%).",
        "r_idx": "Without calibration or a reference plant, greenness cannot be converted to nitrogen %.",
        # verdicts
        "v_noleaf": "No leaf found", "v_retake": "Retake the photo", "v_leaf": "Leaf detected",
        "v_refsaved": "Reference plant saved",
    },
    "ar": {
        "step_quality": "فحص جودة الصورة",
        "step_detect": "اكتشاف الورقة",
        "step_measure": "قياس لون الورقة",
        "step_estimate": "تقدير حالة النيتروجين",
        "step_ref": "حفظ النبات المرجعي",
        "sharp": "الوضوح", "bright": "الإضاءة", "blurry": "غير واضحة", "dark": "مظلمة",
        "covers": "الورقة تغطي {p}٪ من الإطار ({n} منطقة)",
        "no_leaf": "لم يتم العثور على ورقة ({p}٪ بكسلات خضراء)",
        "pixels": "{v}٪ من بكسلات الورقة صالحة، انعكاس {g}٪",
        "skipped": "تم التخطي",
        "mode_cal": "نموذج معاير: {v}٪ نيتروجين",
        "mode_ref": "مقارنة بالمرجع: {si} ← {st}",
        "mode_idx": "تم قياس الاخضرار (بدون معايرة)",
        "ref_saved": "تم حفظ النبات المرجعي (الاخضرار {d})",
        "ref_failed": "لم يتم العثور على ورقة في صورة المرجع",
        "r_quality_ok": "الصورة واضحة ومضاءة بما يكفي للقياس.",
        "r_blurry": "الصورة غير واضحة، لذا لا يمكن قياس اللون بدقة.",
        "r_dark": "الصورة مظلمة جدًا، وقراءات اللون ستكون خاطئة.",
        "r_leaf": "تم العثور على ورقة تغطي {p}٪ من الإطار.",
        "r_small": "الورقة صغيرة في الإطار، لذا تم قياس عدد أقل من البكسلات.",
        "r_noleaf": "أقل من 5٪ من الإطار بلون الأوراق، لذا لم يتم اكتشاف ورقة.",
        "r_glare": "{g}٪ من الورقة عليها انعكاس فلاش وتم استبعادها.",
        "r_dgci": "درجة اخضرار الورقة (DGCI) هي {d}: {word}.",
        "pale": "فاتحة", "medium": "خضراء متوسطة", "dark_green": "خضراء داكنة",
        "r_brown": "{b}٪ من الورقة يبدو بنيًا، وقد يدل ذلك على مرض أو أنسجة قديمة وليس نقص نيتروجين.",
        "r_yellow": "{y}٪ من الورقة يبدو مصفرًا.",
        "r_ref": "مقارنة بنبات المرجع، اخضرار الورقة يساوي {pct}٪ منه (مؤشر الكفاية {si}).",
        "r_ref_rule": "القاعدة: 95٪ فأكثر كافٍ، 90–95٪ حدّي، أقل من 90٪ نقص.",
        "r_cal": "النموذج المعاير يقدّر النيتروجين بـ {v}٪ (المدى {lo}–{hi}٪).",
        "r_idx": "بدون معايرة أو نبات مرجعي، لا يمكن تحويل الاخضرار إلى نسبة نيتروجين.",
        "v_noleaf": "لم يتم العثور على ورقة", "v_retake": "أعد التقاط الصورة", "v_leaf": "تم التعرف على ورقة",
        "v_refsaved": "تم حفظ النبات المرجعي",
    },
}

STATUS_WORD = {
    "en": {"sufficient": "sufficient", "marginal": "borderline", "deficient": "deficient",
           "greener": "greener than reference"},
    "ar": {"sufficient": "كافٍ", "marginal": "حدّي", "deficient": "نقص", "greener": "أكثر اخضرارًا من المرجع"},
}


def suff_key(status: str) -> str:
    if status.startswith("greener"):
        return "greener"
    if status.startswith("deficient"):
        return "deficient"
    if status.startswith("marginal"):
        return "marginal"
    return "sufficient"


def measurement_steps(r: dict, lang: str) -> List[dict]:
    t = T[lang]
    q = r.get("image_quality") or {}
    det = r.get("leaf_detection") or {}
    n = r.get("nitrogen") or {}
    ev = []

    if not q:
        ev.append(dict(id="quality", group="measure", status="error", label=t["step_quality"],
                       detail=", ".join(r.get("reasons", []))))
        return ev
    bad = q["is_blurry"] or q["is_too_dark"]
    detail = f"{t['sharp']} {q['blur_score']:.0f}, {t['bright']} {q['mean_brightness']:.2f}"
    if q["is_blurry"]:
        detail += f" ({t['blurry']})"
    if q["is_too_dark"]:
        detail += f" ({t['dark']})"
    ev.append(dict(id="quality", group="measure", status="warn" if bad else "done",
                   label=t["step_quality"], detail=detail))

    p = f"{det.get('green_fraction', 0) * 100:.0f}"
    if det.get("leaf_detected"):
        k = det.get("n_leaf_regions", 1)
        ev.append(dict(id="detect", group="measure", status="done", label=t["step_detect"],
                       detail=t["covers"].format(p=p, n=k, s="" if k == 1 else "s")))
    else:
        ev.append(dict(id="detect", group="measure", status="error", label=t["step_detect"],
                       detail=t["no_leaf"].format(p=p)))

    if n:
        nq = n.get("quality", {})
        ev.append(dict(id="measure", group="measure", status="done", label=t["step_measure"],
                       detail=t["pixels"].format(v=f"{nq.get('valid_fraction', 0) * 100:.0f}",
                                                 g=f"{nq.get('glare_fraction', 0) * 100:.1f}")))
        if n.get("mode") == "calibrated" and n.get("nitrogen"):
            d = t["mode_cal"].format(v=n["nitrogen"]["value"])
        elif n.get("sufficiency"):
            s = n["sufficiency"]
            d = t["mode_ref"].format(si=s["sufficiency_index_dgci"],
                                     st=STATUS_WORD[lang][suff_key(s["status"])])
        else:
            d = t["mode_idx"]
        ev.append(dict(id="estimate", group="measure", status="done", label=t["step_estimate"], detail=d))
    else:
        for sid, key in (("measure", "step_measure"), ("estimate", "step_estimate")):
            ev.append(dict(id=sid, group="measure", status="skip", label=t[key], detail=t["skipped"]))
    return ev


def dgci_word(d: float, lang: str) -> str:
    t = T[lang]
    return t["pale"] if d < 0.45 else t["medium"] if d < 0.6 else t["dark_green"]


def measurement_reasoning(r: dict, ref: Optional[dict], lang: str) -> List[str]:
    t = T[lang]
    q = r.get("image_quality") or {}
    det = r.get("leaf_detection") or {}
    n = r.get("nitrogen") or {}
    out = []
    if q:
        if q["is_blurry"]:
            out.append(t["r_blurry"])
        if q["is_too_dark"]:
            out.append(t["r_dark"])
        if not (q["is_blurry"] or q["is_too_dark"]):
            out.append(t["r_quality_ok"])
    if det:
        if det.get("leaf_detected"):
            out.append(t["r_leaf"].format(p=f"{det['green_fraction'] * 100:.0f}"))
            if det["green_fraction"] < 0.10:
                out.append(t["r_small"])
        else:
            out.append(t["r_noleaf"])
    if not n:
        return out
    f = n.get("features", {})
    g = (n.get("quality") or {}).get("glare_fraction", 0)
    if g and g > 0.02:
        out.append(t["r_glare"].format(g=f"{g * 100:.0f}"))
    if f.get("dgci") is not None:
        out.append(t["r_dgci"].format(d=f"{f['dgci']:.3f}", word=dgci_word(f["dgci"], lang)))
    if (f.get("frac_brownish") or 0) > 0.05:
        out.append(t["r_brown"].format(b=f"{f['frac_brownish'] * 100:.0f}"))
    elif (f.get("frac_yellowish") or 0) > 0.15:
        out.append(t["r_yellow"].format(y=f"{f['frac_yellowish'] * 100:.0f}"))
    if n.get("mode") == "calibrated" and n.get("nitrogen"):
        nv = n["nitrogen"]
        lo, hi = nv["approx_interval_95"]
        out.append(t["r_cal"].format(v=nv["value"], lo=lo, hi=hi))
    elif n.get("sufficiency"):
        si = n["sufficiency"]["sufficiency_index_dgci"]
        out.append(t["r_ref"].format(pct=f"{si * 100:.0f}", si=si))
        out.append(t["r_ref_rule"])
    else:
        out.append(t["r_idx"])
    return out
