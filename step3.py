#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: code client utterances and split client turns by coding changes.

Rules:
  - First choose segmentation_unit:
    none = keep one transcript line as one coding unit.
    punctuation = split each client line into sentence-like units by punctuation.
  - Therapist utterances are never split or merged; each transcript line stays
    its own segment.
  - Client utterances are coded with behavior, motivation, and time. The LLM
    sees the surrounding conversation (therapist and client lines) as
    read-only context, but only codes the client utterances.
  - Consecutive client utterances are merged only when behavior, motivation,
    and time are all unchanged.
  - Codes follow the MISC-BE manual (MIBE manual.docx). If the model output for
    an utterance is missing or invalid, the utterance is retried; after
    --max_retries it is coded NC (no code) rather than guessed.
  - Motivation second pass (--mot_passes, default 3): every client unit coded
    TB/ALT/OTH is judged again on its own with wider context (--mot_before /
    --mot_after lines) and thinking on, only for motivation. The passes are
    majority-voted; a tie gives "=", and a unit with no valid answer keeps its
    first-pass motivation. Behavior and time are kept. --mot_passes 0 skips it.

Coding Quick Key:
  Behavior:
    th = therapist utterance
    TB = target behavior (alcohol)
    ALT = alternative reinforcer
    OTH = independent reinforcer
    nr = not relevant
    NC = no code (uncodable, or model failed after retries)
  Motivation:
    - = avoid
    = = neutral
    + = approach
  Time:
    P = past
    C = current
    F = future

Only client utterances with behavior TB, ALT, or OTH receive motivation and
time ratings. Client utterances coded nr or NC have motivation/time set to null.
"""

import argparse
import csv
import fnmatch
import glob
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


LABEL_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)\s*$")
SENTENCE_RE = re.compile(r"[^.!?。！？;；]+(?:[.!?。！？;；]+|$)")
ROLE_FROM_LABEL = {
    "therapist": "therapist",
    "patient": "patient",
}
LABEL_FROM_ROLE = {
    "therapist": "therapist",
    "patient": "patient",
}
VALID_BEHAVIORS = {"TB", "ALT", "OTH", "nr", "NC"}
VALID_MOTIVATIONS = {"-", "=", "+"}
VALID_TIMES = {"P", "C", "F"}
UNCODED_BEHAVIORS = {"nr", "NC"}
# Qwen3.8: thinking off for this high-volume step. Sampling follows Qwen's
# non-thinking recommendation, except presence_penalty stays 0 because the
# JSON output repeats the same keys for every utterance.
EXTRA_BODY = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}

SYSTEM_PROMPT = (
    "You are an expert clinical-dialog coder trained on the Motivational "
    "Interviewing Skill Code - Behavioral Economics Version (MISC-BE). "
    "You code client utterances following the coding manual supplied by the user. "
    "Return only valid JSON."
)

# Rules and examples below are taken from the MISC-BE manual (MIBE manual.docx).
USER_PROMPT_TEMPLATE = """\
Code each CLIENT utterance using the MISC-BE coding manual below.
Each client utterance gets three independent decisions: Behavior, Motivation, Time.
The only exception: utterances coded nr or NC get NO motivation and NO time (null).

=====================================================================
1. BEHAVIOR (exactly one of TB / ALT / OTH / nr / NC)
=====================================================================

TB = Target Behavior = ALCOHOL USE.
- The utterance must contain information on the CLIENT'S OWN attitude about or
  engagement in drinking.
- The link to alcohol may be made by the client OR the therapist. When the
  therapist directly references drinking, or links another behavior to
  drinking, code the client's following utterances about it as TB.
- Drinking talked about in general terms ("people", "students") or other
  people's drinking (friends, roommates) is nr, unless the context shows it
  reveals the client's own drinking attitudes or behavior.
- If it is unclear whether the utterance is about the behavior, code nr.
  TB: "I never go out drinking on school nights" / "It's really important to me
      to go out drinking with my friends" / "I feel left out when my friends
      drink without me" / "I can drive after drinking as long as my BAC is
      below the legal limit"
  nr: "My best friend loves partying on the weekends" / "Most students here
      only drink on weekends" / "The legal driving limit for BAC in this state is 0.08"

ALT = Alternative behavior: competes with drinking for limited resources
(money, time, legal costs, health). Automatically ALT:
  physical activity / exercise / sports; diet / weight management;
  academic activities (studying, attending class, grades, standardized tests
  such as GRE/MCAT/LSAT, employment opportunities as a result of academic
  success, discussions of their major).
- Any OTHER behavior is ALT only if the client or therapist directly links it
  as competing with drinking (e.g. client said they never drink with family;
  later utterances about valuing family time become ALT). Otherwise it is OTH.
- General college engagement without explicit academics is OTH.
  "Doing well in school will help me get a job I like" (ALT, C, +)
  "I really like college" (OTH, C, +)
  "For me, my schoolwork always comes first" (ALT) / "I want to start exercising more" (ALT)

OTH = Independent behavior: an overt behavior the client can allocate time or
resources to, not clearly linked to drinking. Social life (family, friends,
partners), jobs, internships, clubs, hobbies are OTH unless explicitly linked
to drinking (then TB) or to competing with drinking (then ALT).
- Merely expressing a preference that is not about a behavior is nr, not OTH.
  "I really like hanging out with my friends" (OTH, C, +)
  "I've met so many friends while out drinking at parties" (TB, C, +)
  "My friends and I work out at the Rec Center together" (ALT, C, =)
  "I work part-time at an ice cream shop" (OTH, C, =)
  "I'm not sure if I can get an internship until my senior year" (OTH, F, =)
  "I like to play board games" (OTH) vs "I don't like rainy days" (nr)

nr = Not relevant. Use for utterances that:
- do not refer to a specific behavioral act (small talk, clarifying questions);
- are incomplete or partially inaudible;
- only show the client is listening/tracking without adding content or opinion;
- are about others' behavior or the behavior in general terms.
  nr: "My best friend used to drink before school" / "I've never been outside
      of the state" / "I must have answered a question wrong on the survey" /
      "I am stressed out"

NC = No code. Use RARELY, only when the utterance is uncodable: transcript is
garbled/unintelligible for most of the utterance, a third person is speaking,
a phone call, or a pause in the session (e.g. fire alarm).

Behavior decision rules:
- Multiple behaviors in one utterance: use the hierarchy TB > ALT > OTH.
  "When I drink, I don't feel like going to the gym in the morning" (TB, C, -)
  "Being hungover makes it difficult to pay attention in class" (TB, C, -)
  "Studying is more important than going to the movies with friends" (ALT, C, +)
- Marijuana and other drugs are OTH, unless used together with alcohol (TB) or
  established as competing with alcohol (ALT).
  "I like smoking marijuana" (OTH, C, +) / "I only use marijuana while drinking"
  (TB, C, =) / "I use Adderall to help me study" (ALT, C, +)
- When in doubt between ALT and OTH, code OTH unless it is one of the
  automatically-ALT behaviors listed above.
- Facilitative replies: "Yeah" / "Uh-huh" / "Okay" that only signal listening
  are nr. If the reply is active agreement/disagreement with a statement about
  the client's own behavior, or adds elaboration, code it accordingly.
  T: "The BAC driving limit is 0.08."  C: "Yeah" (nr)
  T: "You mentioned that you get pretty bad hangovers from alcohol."  C: "Yeah" (TB, C, -)
- Numerical ratings: a bare number answer to a rating question is nr. If the
  client explains the number, code the content.
  T: "On a scale of 1-10, how important is studying?"  C: "Seven." (nr)
  C: "Seven, because studying is really important for getting good grades" (ALT, C, +)
- If one unit contains several utterances, code the LAST complete utterance.

Codable units that are often wrongly coded nr:
- Units are sentence pieces: a client turn was split at sentence punctuation,
  so one thought can span several units. A unit that continues or completes
  the client's own thought ("Always your reaction time.", "and not really
  anything.", "Like on weekends.") is NOT incomplete: code it by the thought
  it belongs to, using the neighbouring client units and the following lines.
  The same holds for a unit that STARTS a thought the next units complete
  ("I mean, I guess to some extent, like, I...", "And so, yeah, and she...").
  Use nr for "incomplete" only when the unit has no codable content even
  together with its neighbours.
- Stories about the client's own drinking. When the client is telling what
  happened on a night they drank (or a drinking situation they were in), every
  unit of the story is TB, including what other people did or said to them
  ("my friend came over and started screaming at me", "my boyfriend just took me
  home"). Code its motivation by what the story shows about their drinking.
  The same applies to stories about the client's own ALT/OTH behavior.
  This applies ONLY to stories about a drinking night or drinking situation.
  Other personal topics (the client's child, family, health, relationships)
  are not TB just because drinking came up earlier in the session.
- Self-descriptions given to explain the client's own drinking ("I'm the kind
  of person who likes to get crazy", "I'm an outgoing guy who likes to go
  out", "it's just me and my group of friends") are TB. Reactions to the
  client's OWN drinking feedback (their percentile, their BAC, their binge
  nights: "I don't believe that", "that's weird to me", "that seems high") are
  TB, not nr. General statements phrased as "no one / you / people" that the
  context shows are the client's own attitude ("No one wants to be the
  annoying drunk") are TB.
- Short answers. "Yes", "Yeah", "No", "Not really", "Probably", "I think so"
  that ANSWER a therapist question, or CONFIRM/DENY a therapist reflection,
  about the client's OWN drinking or other behavior are coded by the content
  they confirm or deny (T: "Do you think you could stick to three drinks?"
  C: "Yes." -> TB, F, -).
- These stay nr (manual): replies that only show listening ("Okay",
  "Uh-huh", "Right" while the therapist explains); acknowledging information
  (statistics, norms, definitions, BAC charts, handouts) without saying
  anything about the client's own behavior or attitude; "I don't know" with no
  content; answers to small talk or procedural questions; bare numbers in
  answer to a rating question; facts or opinions about other people or
  students in general that say nothing about the client; explanations of how
  the client filled in the survey or estimated other students' drinking that
  give no information about their own drinking or attitude; questions the
  client asks the therapist; "No" / "I don't think so" in answer to "Any
  questions?".
- Career plans, which college to attend, coping skills, and money management
  are OTH (not automatic ALT) unless linked to drinking. The client's own
  classes, studying, grades, exercise and sports (running, biking, team
  practice) remain automatic ALT.

=====================================================================
2. MOTIVATION (only for TB / ALT / OTH)
=====================================================================
Motivation is about the reinforcement value of THE BEHAVIOR BEING DISCUSSED.

"-" Avoid: lower likelihood of the behavior because its value is reduced:
  adverse/undesired consequences of the behavior, positive consequences of
  abstaining or reducing it, negative attitudes toward doing it, or positive
  attitudes toward doing less of it.
  "My hangovers are terrible." (TB, C, -)
  "It was really scary when I woke up and didn't remember anything" (TB, P, -)
  "I am hoping to reduce my drinking from 10 drinks to 4" (TB, C, -)
  "It's so difficult to find motivation to go to the gym." (ALT, C, -)
  "I don't need to study to get an A in the class." (ALT, C, -)
- Undesirable consequences are avoid unless the client explicitly states a
  competing opinion about the consequence.
- Acknowledged negative consequences of an otherwise approached behavior are
  avoid at the utterance level.

"+" Approach: higher likelihood of the behavior because of its value:
  positive/desired consequences, pleasurable outcomes, barriers or adverse
  consequences of cutting down/stopping, or a modified action/belief that
  allows the behavior to continue.
  "I like alcohol because it helps me to be more social." (TB, C, +)
  "I never get sick or have hangovers." (TB, C, +)
  "Cutting down drinking would make it hard to see my friends." (TB, C, +)
  "I don't want to get too drunk and go overboard" (TB, C, -) "so I'll only
   have a few drinks" (TB, C, +)
  "I'm not concerned about my drinking because I know my limits" (TB, C, +)
  "I won't stay out late so I can pay attention in my morning class" (ALT, C, +)

"=" Neutral: no direction, ambiguous value, or a non-valenced fact/report.
Neutral is the DEFAULT when meaning is unclear.
  "Alcohol makes me drunk." (TB, C, =) / "I drink three times a week." (TB, C, =)
  "I am surprised that I drink more than other people my age" (TB, C, =)
- Equal approach and avoid content -> "=".
  "I want to exercise three times a week, but I hate feeling sore afterwards." (ALT, C, =)
- Neutral mixed with approach or avoid -> follow the overall direction.
  "Alcohol usually makes me feel buzzed. I like that feeling." (TB, C, +)

Deciding motivation from context:
- Read the utterance in context, not in isolation. Ask: what does it do for
  the behavior being coded - make it more likely (+), less likely (-), or
  neither (=)? Use the therapist's question and the client's neighbouring
  units, including the following lines.
- For TB, whenever the client expresses an attitude, intention, reason,
  feeling, or evaluation that favors drinking or keeping their drinking as it
  is, code +; whenever they express one that favors drinking less, stopping,
  or a negative view of their drinking or its consequences, code -. Do not
  default to "=" for these.
- Answers inherit the direction of the question. Units that answer "What do
  you like about drinking / what are the good things?" are +; units that
  answer "What don't you like / what would be the benefits of cutting down?"
  are - (unless the unit itself says otherwise). Items in a list share the
  direction of the list.
- Minimizing, denying problems, or dismissing concern is APPROACH (+) for TB:
  "I can handle it", "I know my limits so I'm not worried", "that's the
  absolute max", "it never happens to me", "it's always fine", "the
  definition doesn't apply to me", "I don't believe that number", "it's easy
  to get drinks", "I drink more when I've eaten".
- Drinking LESS is AVOID (-) for TB: reports that the client drinks less than
  before or has not been drinking ("not as much lately", "I haven't drunk",
  "I don't drink to that level anymore"), and anything the client does to
  drink less or avoid heavy drinking: limits, slowing down or stopping,
  not going out on certain nights, leaving, resisting pressure, choosing not
  to drink, avoiding drink-driving or risky situations ("I'm not one of those
  people who goes out on Thursdays", "I'll slow down in the middle", "I've
  never felt pressure to drink more", "I just leave").
- Costs, effort or barriers of CUTTING DOWN are APPROACH (+) for TB ("it
  takes a lot of effort to stay under my limit", "buying soda instead would
  be another extra cost", "my friends wouldn't hang out with me").
- For ALT/OTH the same logic applies to that behavior: liking, valuing,
  trying, doing it well, or planning to do more of it is +; doing less of it,
  not doing it, doing it badly, not caring about it, or barriers to doing it
  are - ("I don't study a lot at all" -> ALT, C, -; "I normally try to study
  between classes" -> ALT, C, +; "my schedule leaves no time to work out" ->
  ALT, C, -).
- A short reply ("Yes", "Yeah, for sure", "No") takes the direction of the
  statement it confirms; a "No" that denies a statement takes the opposite
  direction.
- "=" is for plain reports with no attitude, reason, or change ("I drink
  three times a week" answering "How often do you drink?", "Alcohol makes me
  drunk", "I have class at 9"), and for units where approach and avoid are
  equally present.

=====================================================================
3. TIME (only for TB / ALT / OTH)
=====================================================================
Time is WHEN the reinforcing value (reward or punishment) is earned or
received, not only when the behavior happens.

C Current = DEFAULT.
- Current consequences, currently held attitudes without an explicit
  past/future reason, unspecified timing, and anything within the past week or
  expected in the week after the session.
- Behavior patterns (molar, habitual statements) are C.
- When multiple time frames apply, default to C.
  T: "What happened the last time you drank?"  C: "Oh that was a few days
  ago... I went to an amazing party" (TB, C, +)
  "I tend to drink heavier during the summers." (TB, C, =)
  "I have always been a good student." (ALT, C, +)
  "Yeah, but I don't drink to that level anymore." (TB, C, -)

P Past = attitude held in the past, or consequences already experienced
(more than a week ago), stated in past tense or clear from context. If a past
consequence is explicitly given as a reason for current behavior, code C.
  T: "And this is 10 drinks in 8 hours."  C: "Yeah...that was game day." (TB, P, =)

F Future = behavior or consequences projected to be experienced in the
future, including hypothetical consequences not yet experienced.
Attitudes are C unless explicitly explained by expected future consequences.
  T: "How would you know when it becomes too much?"  C: "When I spend more time
  drinking than studying." (TB, F, -)
  "I want to get a good GPA." (ALT, C, +) -> "Because it will help me be
  successful and make lots of money." (ALT, F, +)

=====================================================================
INPUT
=====================================================================
Therapist lines are context only. Use them (and nearby client lines) to
understand what a client utterance refers to, e.g. whether "Yeah" is listening
or agreement, or whether the therapist has linked a behavior to drinking.
Do NOT code therapist lines.

Conversation context (read-only, in chronological order):
{context}

Following lines (read-only, NOT to be coded; use them to see how a thought or
answer continues):
{after}

Client utterances to code:
{utterances}

=====================================================================
OUTPUT
=====================================================================
Return JSON only, no markdown fences, with exactly this shape. Write the short
reason in "evidence" FIRST, then the codes:
{{
  "utterances": [
    {{
      "utt_id": 0,
      "evidence": "short reason citing the manual rule (do not use double quotes inside it)",
      "behavior": "TB",
      "motivation": "+",
      "time": "C"
    }}
  ]
}}

Rules:
- Return exactly one entry for every client utt_id listed under "Client utterances to code".
- behavior must be one of TB, ALT, OTH, nr, NC.
- For TB/ALT/OTH, motivation must be one of "-", "=", "+" and time one of P, C, F.
- For nr or NC, motivation and time must be null.
"""

# Second pass: re-decide motivation for one TB/ALT/OTH unit at a time.
MOT_EXTRA_BODY = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
MANUAL_MOTIVATION = USER_PROMPT_TEMPLATE[
    USER_PROMPT_TEMPLATE.index("2. MOTIVATION"):USER_PROMPT_TEMPLATE.index("Deciding motivation from context:")
][:-1]
MOTIVATION_RULES = USER_PROMPT_TEMPLATE[
    USER_PROMPT_TEMPLATE.index("Deciding motivation from context:"):USER_PROMPT_TEMPLATE.index("\n====", USER_PROMPT_TEMPLATE.index("Deciding motivation from context:"))
]
MOT_BEHAVIOR_NAMES = {
    "TB": "target behavior: the client's own drinking/drug use",
    "ALT": "alternative behavior: academics, studying, exercise, health activities",
    "OTH": "other independent behavior",
}
MOT_PROMPT_TEMPLATE = """\
You are coding client speech in a motivational interviewing session with the
MISC-BE manual. The client unit marked >> has already been coded with behavior
{behavior} ({bname}). Decide only its MOTIVATION: +, -, or =.

{manual}

{rules}
Procedure:
1. Say what the unit means in context: what the therapist asked, and what the
   client is saying in this and the neighbouring units (a fragment continues
   the client's previous sentence).
2. Say whether, for the {behavior} behavior, the unit makes it more likely or more
   valued (+), less likely or less valued (-), or neither / equally both (=).
3. Give the answer.

Conversation ([T] therapist, [C] client):
{context}

Answer with JSON only:
{{"reasoning": "2-4 sentences, no double quotes", "motivation": "+|-|="}}
"""
MOT_RE = re.compile(r'"motivation"\s*:\s*"([+=-])"')


def normalize_role(label: str) -> Optional[str]:
    return ROLE_FROM_LABEL.get(label.strip().lower()) or ROLE_FROM_LABEL.get(label.strip())


def parse_corrected_txt(path: str) -> List[Dict[str, object]]:
    utterances: List[Dict[str, object]] = []
    previous_role: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            match = LABEL_RE.match(line)
            if match:
                role = normalize_role(match.group(1)) or previous_role or "patient"
                text = match.group(2).strip()
            else:
                role = previous_role or "patient"
                text = line
            if not text:
                continue
            utterances.append({
                "utt_id": len(utterances),
                "line_no": line_no,
                "role": role,
                "label": LABEL_FROM_ROLE[role],
                "text": text,
            })
            previous_role = role
    return utterances


def parse_corrected_csv(path: str) -> List[Dict[str, object]]:
    utterances: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = normalize_role(str(row.get("role", ""))) or "patient"
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            try:
                utt_id = int(row.get("utt_id", len(utterances)))
            except Exception:
                utt_id = len(utterances)
            try:
                line_no = int(row.get("line_no", utt_id + 1))
            except Exception:
                line_no = utt_id + 1
            utterances.append({
                "utt_id": utt_id,
                "line_no": line_no,
                "role": role,
                "label": LABEL_FROM_ROLE[role],
                "text": text,
            })
    return utterances


def parse_corrected_input(path: str) -> List[Dict[str, object]]:
    if path.lower().endswith(".csv"):
        return parse_corrected_csv(path)
    return parse_corrected_txt(path)


def split_text_by_punctuation(text: str) -> List[str]:
    parts = [match.group(0).strip() for match in SENTENCE_RE.finditer(text)]
    parts = [part for part in parts if part]
    return parts or [text.strip()]


def split_utterances(utterances: List[Dict[str, object]], segmentation_unit: str) -> List[Dict[str, object]]:
    if segmentation_unit == "none":
        for idx, row in enumerate(utterances):
            row["unit_id"] = idx
            row["source_utt_id"] = row["utt_id"]
            row["source_unit_index"] = 0
        return utterances

    split_rows: List[Dict[str, object]] = []
    for row in utterances:
        if row["role"] == "therapist":
            parts = [str(row["text"]).strip()]
        else:
            parts = split_text_by_punctuation(str(row["text"]))
        for unit_index, part in enumerate(parts):
            split_rows.append({
                "utt_id": len(split_rows),
                "unit_id": len(split_rows),
                "source_utt_id": row["utt_id"],
                "source_unit_index": unit_index,
                "line_no": row["line_no"],
                "role": row["role"],
                "label": row["label"],
                "text": part,
            })
    return split_rows


def extract_json(text: str) -> Optional[Dict]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except Exception:
        return salvage_codes(cleaned)


ITEM_RE = re.compile(r'"utt_id"\s*:\s*(\d+)(.*?)(?="utt_id"\s*:|\Z)', re.S)


def salvage_codes(text: str) -> Optional[Dict]:
    """Recover per-utterance codes from almost-valid JSON (e.g. an unescaped
    quote inside "evidence"), so one bad character does not lose a batch."""
    items = []
    for match in ITEM_RE.finditer(text):
        body = match.group(2)
        fields = {}
        for name in ("behavior", "motivation", "time"):
            found = re.search(rf'"{name}"\s*:\s*(null|"([^"]*)")', body)
            if found:
                fields[name] = found.group(2) if found.group(2) is not None else None
        if "behavior" not in fields:
            continue
        evidence = re.search(r'"evidence"\s*:\s*"(.*?)",\s*"(?:behavior|motivation|time)"', body, re.S)
        fields["evidence"] = evidence.group(1) if evidence else ""
        items.append({"utt_id": int(match.group(1)), **fields})
    return {"utterances": items} if items else None


def normalize_behavior(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    upper = value.upper()
    if upper in {"TB", "ALT", "OTH"}:
        return upper
    if value.lower() in {"nr", "not relevant", "not_relevant"}:
        return "nr"
    if upper in {"NC", "NO CODE", "NO_CODE"}:
        return "NC"
    return None


def normalize_motivation(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    aliases = {
        "avoid": "-",
        "negative": "-",
        "neutral": "=",
        "approach": "+",
        "positive": "+",
    }
    value = aliases.get(value, value)
    return value if value in VALID_MOTIVATIONS else None


def normalize_time(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    aliases = {
        "PAST": "P",
        "CURRENT": "C",
        "PRESENT": "C",
        "FUTURE": "F",
    }
    value = aliases.get(value, value)
    return value if value in VALID_TIMES else None


def failed_code(reason: str) -> Dict[str, object]:
    """Code used when the model gives no valid code after all retries."""
    return {
        "behavior": "NC",
        "motivation": None,
        "time": None,
        "evidence": f"model_failed: {reason}",
    }


def format_client_utterances(batch: List[Dict[str, object]]) -> str:
    return "\n".join(f'{row["utt_id"]}: {row["text"]}' for row in batch)


def format_context(rows: List[Dict[str, object]]) -> str:
    return "\n".join(f'{row["utt_id"]}: [{row["role"]}] {row["text"]}' for row in rows)


def call_llm(
    client: OpenAI,
    model: str,
    context_rows: List[Dict[str, object]],
    batch: List[Dict[str, object]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    after_rows: Optional[List[Dict[str, object]]] = None,
) -> Tuple[Optional[Dict], str, float]:
    prompt = USER_PROMPT_TEMPLATE.format(
        context=format_context(context_rows),
        utterances=format_client_utterances(batch),
        after=format_context(after_rows or []) or "(end of session)",
    )
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body=EXTRA_BODY,
    )
    latency = round(time.time() - t0, 3)
    raw = response.choices[0].message.content or ""
    return extract_json(raw), raw, latency


def normalize_code(item: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Return a valid code, or None if the model output is invalid."""
    behavior = normalize_behavior(item.get("behavior"))
    if behavior is None:
        return None
    evidence = str(item.get("evidence") or "").strip()
    if behavior in UNCODED_BEHAVIORS:
        return {
            "behavior": behavior,
            "motivation": None,
            "time": None,
            "evidence": evidence,
        }
    motivation = normalize_motivation(item.get("motivation"))
    code_time = normalize_time(item.get("time"))
    if motivation is None or code_time is None:
        return None
    return {
        "behavior": behavior,
        "motivation": motivation,
        "time": code_time,
        "evidence": evidence,
    }


def codes_from_parsed(parsed: Optional[Dict], batch: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    """Return valid codes only; utterances missing or invalid are left out."""
    target_ids = {int(row["utt_id"]) for row in batch}
    codes: Dict[int, Dict[str, object]] = {}
    if not isinstance(parsed, dict):
        return codes
    for item in parsed.get("utterances", []):
        if not isinstance(item, dict):
            continue
        try:
            utt_id = int(item.get("utt_id"))
        except Exception:
            continue
        if utt_id not in target_ids:
            continue
        code = normalize_code(item)
        if code is not None:
            codes[utt_id] = code
    return codes


def code_client_utterances(
    utterances: List[Dict[str, object]],
    coding_mode: str,
    model: Optional[str],
    openai_api_base: str,
    api_key: str,
    batch_size: int,
    context_utts: int,
    max_retries: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    after_utts: int = 3,
) -> Tuple[Dict[int, Dict[str, object]], List[Dict[str, object]]]:
    if coding_mode != "llm":
        raise ValueError(f"Unsupported coding_mode: {coding_mode}")
    if not model:
        raise ValueError("--model is required")

    patient_rows = [row for row in utterances if row["role"] == "patient"]
    pos_by_id = {int(row["utt_id"]): pos for pos, row in enumerate(utterances)}
    codes: Dict[int, Dict[str, object]] = {}
    logs: List[Dict[str, object]] = []

    client = OpenAI(api_key=api_key, base_url=openai_api_base)
    for start in range(0, len(patient_rows), batch_size):
        batch = patient_rows[start:start + batch_size]
        batch_id = start // batch_size
        # Context = all lines (therapist + client) from context_utts lines before
        # the first target client utterance through the last one in this batch.
        first_pos = pos_by_id[int(batch[0]["utt_id"])]
        last_pos = pos_by_id[int(batch[-1]["utt_id"])]
        context_rows = utterances[max(0, first_pos - context_utts):last_pos + 1]
        # Read-only lines after the batch, to see how a thought or answer continues.
        after_rows = utterances[last_pos + 1:last_pos + 1 + after_utts]

        # Retry only the utterances that are still missing a valid code.
        pending = batch
        last_error = "no valid code in model output"
        for attempt in range(max_retries + 1):
            try:
                parsed, raw_output, latency = call_llm(
                    client=client,
                    model=model,
                    context_rows=context_rows,
                    batch=pending,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout,
                    after_rows=after_rows,
                )
            except Exception as exc:
                parsed, raw_output, latency = None, repr(exc), 0.0
                last_error = f"request error: {exc}"
                print(f"[ERR ] coding batch={batch_id} attempt={attempt} {exc}")

            batch_codes = codes_from_parsed(parsed, pending)
            codes.update(batch_codes)
            pending = [row for row in pending if int(row["utt_id"]) not in batch_codes]
            print(
                f"[CODE] batch={batch_id} attempt={attempt} "
                f"patient_utt={batch[0]['utt_id']}-{batch[-1]['utt_id']} "
                f"latency={latency}s coded={len(batch_codes)} still_missing={len(pending)}"
            )
            logs.append({
                "batch_id": batch_id,
                "attempt": attempt,
                "start_utt_id": batch[0]["utt_id"],
                "end_utt_id": batch[-1]["utt_id"],
                "latency_sec": latency,
                "parsed_ok": parsed is not None,
                "parsed_json": parsed,
                "raw_output": raw_output,
            })
            if not pending:
                break

        for row in pending:
            codes[int(row["utt_id"])] = failed_code(last_error)
        if pending:
            print(f"[WARN] batch={batch_id} {len(pending)} utterances coded NC after {max_retries + 1} attempts")

    return codes, logs


def build_motivation_prompt(utterances: List[Dict[str, object]], pos: int, behavior: str, before: int, after: int) -> str:
    tag = lambda row: "C" if row["role"] == "patient" else "T"  # noqa: E731
    lines = [f"  [{tag(r)}] {r['text']}" for r in utterances[max(0, pos - before):pos]]
    lines.append(f">>[C] {utterances[pos]['text']}")
    lines += [f"  [{tag(r)}] {r['text']}" for r in utterances[pos + 1:pos + 1 + after]]
    return MOT_PROMPT_TEMPLATE.format(
        behavior=behavior,
        bname=MOT_BEHAVIOR_NAMES[behavior],
        manual=MANUAL_MOTIVATION,
        rules=MOTIVATION_RULES,
        context="\n".join(lines),
    )


def ask_motivation(client: OpenAI, model: str, prompt: str, temperature: float, top_p: float,
                   max_tokens: int, timeout: float, tries: int = 3) -> Tuple[Optional[str], str]:
    raw = ""
    for _ in range(tries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body=MOT_EXTRA_BODY,
            )
            raw = response.choices[0].message.content or ""
            found = MOT_RE.findall(raw)
            if found:
                return found[-1], raw
        except Exception as exc:  # noqa: BLE001
            raw = f"request error: {exc}"
    return None, raw


def refine_motivation(
    utterances: List[Dict[str, object]],
    codes: Dict[int, Dict[str, object]],
    model: str,
    openai_api_base: str,
    api_key: str,
    passes: int,
    before: int,
    after: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    workers: int,
) -> List[Dict[str, object]]:
    """Second pass: majority vote over `passes` independent motivation judgments.

    Changes codes in place and returns one log entry per refined unit.
    """
    if passes <= 0:
        return []
    client = OpenAI(api_key=api_key, base_url=openai_api_base)
    targets = [
        (pos, int(row["utt_id"]))
        for pos, row in enumerate(utterances)
        if row["role"] == "patient" and codes.get(int(row["utt_id"]), {}).get("behavior") in MOT_BEHAVIOR_NAMES
    ]
    jobs = [
        (uid, build_motivation_prompt(utterances, pos, codes[uid]["behavior"], before, after))
        for pos, uid in targets
        for _ in range(passes)
    ]
    print(f"[MOT ] second pass: {len(targets)} units x {passes} passes")

    def work(job):
        uid, prompt = job
        return uid, ask_motivation(client, model, prompt, temperature, top_p, max_tokens, timeout)

    votes: Dict[int, List[str]] = {uid: [] for _, uid in targets}
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for uid, (mot, _raw) in pool.map(work, jobs):
            if mot:
                votes[uid].append(mot)

    logs, changed, failed = [], 0, 0
    for _, uid in targets:
        code, v = codes[uid], votes[uid]
        old = code["motivation"]
        if not v:
            failed += 1
            new = old
        else:
            top = Counter(v).most_common()
            new = "=" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]
        if new != old:
            changed += 1
            code["evidence"] = f"[2nd pass vote {''.join(v)}: {old}->{new}] " + str(code.get("evidence") or "")
            code["motivation"] = new
        logs.append({"utt_id": uid, "behavior": code["behavior"], "first_pass": old, "votes": "".join(v), "final": new})
    print(f"[MOT ] changed={changed} no_valid_answer={failed}")
    return logs


def code_key(role: str, code: Optional[Dict[str, object]]) -> Tuple[object, object, object]:
    if role == "therapist":
        return ("th", None, None)
    if not code:
        return ("NC", None, None)
    return (code.get("behavior"), code.get("motivation"), code.get("time"))


def display_code(role: str, code: Optional[Dict[str, object]]) -> str:
    if role == "therapist":
        return "th"
    behavior, motivation, code_time = code_key(role, code)
    if behavior in UNCODED_BEHAVIORS:
        return str(behavior)
    return f"{behavior}{motivation}{code_time}"


def build_segments(utterances: List[Dict[str, object]], codes: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
    return build_segments_with_policy(utterances, codes, merge_patient_same_code=True)


def build_segments_with_policy(
    utterances: List[Dict[str, object]],
    codes: Dict[int, Dict[str, object]],
    merge_patient_same_code: bool,
) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    for utt in utterances:
        role = str(utt["role"])
        utt_id = int(utt["utt_id"])
        code = None if role == "therapist" else codes.get(utt_id, failed_code("no code produced"))
        key = code_key(role, code)

        # Therapist utterances are never merged; only consecutive client
        # utterances with identical behavior/motivation/time are merged.
        can_merge = (
            merge_patient_same_code
            and role == "patient"
            and bool(segments)
            and segments[-1]["role"] == "patient"
            and tuple(segments[-1]["code_key"]) == key
        )
        if can_merge:
            segments[-1]["text"] = f'{segments[-1]["text"]} {utt["text"]}'.strip()
            segments[-1]["source_utt_ids"].append(utt_id)
            segments[-1]["source_line_nos"].append(utt["line_no"])
            if role == "patient":
                segments[-1]["utterance_codes"].append({
                    "utt_id": utt_id,
                    **code,
                })
            continue

        segment = {
            "segment_id": len(segments),
            "role": role,
            "label": LABEL_FROM_ROLE[role],
            "code": display_code(role, code),
            "code_key": list(key),
            "text": str(utt["text"]),
            "source_utt_ids": [utt_id],
            "source_line_nos": [utt["line_no"]],
        }
        if role == "patient":
            segment["behavior"] = code["behavior"]
            segment["motivation"] = code["motivation"]
            segment["time"] = code["time"]
            segment["utterance_codes"] = [{
                "utt_id": utt_id,
                **code,
            }]
        else:
            segment["behavior"] = "th"
            segment["motivation"] = None
            segment["time"] = None
        segments.append(segment)
    return segments


def output_path(input_path: str, input_dir: str, out_dir: str) -> str:
    rel_dir = os.path.relpath(os.path.dirname(input_path), input_dir)
    parent = os.path.join(out_dir, rel_dir)
    os.makedirs(parent, exist_ok=True)
    base = os.path.basename(input_path)
    stem = base.replace(".speaker_corrected.txt", "")
    stem = stem.replace(".speaker_corrected.csv", "")
    return os.path.join(parent, f"{stem}.coded_segments.csv")


def write_outputs(
    out_csv: str,
    input_txt: str,
    utterances: List[Dict[str, object]],
    segments: List[Dict[str, object]],
    coding_mode: str,
    segmentation_unit: str,
    coding_logs: List[Dict[str, object]],
) -> None:
    fieldnames = [
        "input_file", "coding_mode", "segmentation_unit",
        "segment_id", "role", "code", "behavior", "motivation", "time",
        "text", "source_utt_ids", "source_line_nos", "utterance_codes",
    ]
    tmp = out_csv + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            writer.writerow({
                "input_file": input_txt,
                "coding_mode": coding_mode,
                "segmentation_unit": segmentation_unit,
                "segment_id": segment.get("segment_id", ""),
                "role": segment.get("role", ""),
                "code": segment.get("code", ""),
                "behavior": segment.get("behavior", ""),
                "motivation": segment.get("motivation", ""),
                "time": segment.get("time", ""),
                "text": segment.get("text", ""),
                "source_utt_ids": json.dumps(segment.get("source_utt_ids", []), ensure_ascii=False),
                "source_line_nos": json.dumps(segment.get("source_line_nos", []), ensure_ascii=False),
                "utterance_codes": json.dumps(segment.get("utterance_codes", []), ensure_ascii=False),
            })
    os.replace(tmp, out_csv)


def filter_files(files: List[str], include: Optional[List[str]], exclude: Optional[List[str]]) -> List[str]:
    if include:
        files = [
            f for f in files
            if any(fnmatch.fnmatch(f, pat) or pat in f for pat in include)
        ]
    if exclude:
        files = [
            f for f in files
            if not any(fnmatch.fnmatch(f, pat) or pat in f for pat in exclude)
        ]
    return files


def parse_args():
    parser = argparse.ArgumentParser(description="Split speaker-corrected transcripts by Quick Key coding changes")
    parser.add_argument("--input_dir", default="speaker_corrected_txt")
    parser.add_argument("--glob", default="**/*.speaker_corrected.csv")
    parser.add_argument("--out_dir", default="speaker_merged_turns")
    parser.add_argument(
        "--segmentation_unit",
        choices=["none", "punctuation"],
        default="punctuation",
        help="punctuation (default) splits each client line by sentence-ending punctuation before coding; none keeps transcript lines intact",
    )
    parser.add_argument("--coding_mode", choices=["llm"], default="llm")
    parser.add_argument("--model", default=None, help="Model id served by vLLM/OpenAI-compatible API")
    parser.add_argument("--openai_api_base", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="token-abc123")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--context_utts",
        type=int,
        default=10,
        help="Number of preceding transcript lines (therapist + client) to include as context before each batch",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Retries for utterances with missing/invalid model codes; still-failed utterances are coded NC",
    )
    parser.add_argument(
        "--after_utts",
        type=int,
        default=3,
        help="Number of following transcript lines shown read-only after each batch",
    )
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--mot_passes", type=int, default=3,
                        help="Motivation second-pass judgments per TB/ALT/OTH unit (majority vote); 0 disables")
    parser.add_argument("--mot_before", type=int, default=15, help="Lines before the unit in the second pass")
    parser.add_argument("--mot_after", type=int, default=5, help="Lines after the unit in the second pass")
    parser.add_argument("--mot_temperature", type=float, default=0.6)
    parser.add_argument("--mot_top_p", type=float, default=0.95)
    parser.add_argument("--mot_max_tokens", type=int, default=6000)
    parser.add_argument("--mot_workers", type=int, default=32, help="Parallel second-pass requests")
    parser.add_argument("--include", nargs="+", default=None)
    parser.add_argument("--exclude", nargs="+", default=None)
    parser.add_argument("--range", nargs=2, type=int, default=None, metavar=("START", "END"))
    parser.add_argument("--max_files", type=int, default=0, help="Maximum files to process; 0 means no limit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob), recursive=True))
    files = filter_files(files, args.include, args.exclude)
    if args.range:
        files = files[args.range[0]:args.range[1]]
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise RuntimeError(f"No speaker-corrected csv files found under: {args.input_dir}")

    print(f"[INFO] Found {len(files)} speaker-corrected csv files")
    print(f"[INFO] coding_mode: {args.coding_mode}")
    print(f"[INFO] segmentation_unit: {args.segmentation_unit}")
    for path in files:
        print(f"\n[FILE] {path}")
        out_csv = output_path(path, args.input_dir, args.out_dir)
        raw_utterances = parse_corrected_input(path)
        utterances = split_utterances(raw_utterances, args.segmentation_unit)
        codes, coding_logs = code_client_utterances(
            utterances=utterances,
            coding_mode=args.coding_mode,
            model=args.model,
            openai_api_base=args.openai_api_base,
            api_key=args.api_key,
            batch_size=args.batch_size,
            context_utts=args.context_utts,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            after_utts=args.after_utts,
        )
        mot_logs = refine_motivation(
            utterances=utterances,
            codes=codes,
            model=args.model,
            openai_api_base=args.openai_api_base,
            api_key=args.api_key,
            passes=args.mot_passes,
            before=args.mot_before,
            after=args.mot_after,
            temperature=args.mot_temperature,
            top_p=args.mot_top_p,
            max_tokens=args.mot_max_tokens,
            timeout=args.timeout,
            workers=args.mot_workers,
        )
        if mot_logs:
            with open(out_csv.replace(".coded_segments.csv", ".motivation_votes.json"), "w", encoding="utf-8") as f:
                json.dump(mot_logs, f, ensure_ascii=False, indent=1)
        segments = build_segments_with_policy(
            utterances=utterances,
            codes=codes,
            merge_patient_same_code=True,
        )
        write_outputs(
            out_csv=out_csv,
            input_txt=path,
            utterances=utterances,
            segments=segments,
            coding_mode=args.coding_mode,
            segmentation_unit=args.segmentation_unit,
            coding_logs=coding_logs,
        )
        print(f"[OUT ] {out_csv}")


if __name__ == "__main__":
    main()
