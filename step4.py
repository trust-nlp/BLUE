#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified session-component segmentation runner for AAAA-BLUE2+3.

For each Step 3 *.coded_segments.csv file, this program chooses the BMI/BAI or
SFAS component prompt by path unless --session_type overrides it.
"""

import argparse
import asyncio
import csv
import fnmatch
import glob
import json
import os
import time
import re
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI


DEFAULT_CLEAN_FIELDS = [
    "segment_id",
    "role",
    "behavior",
    "motivation",
    "time",
    "text",
    "session_component",
]

BMI_SEGMENT_PROMPT = r"""
You are an expert analyst of Motivational Interviewing (MI) sessions.

Your task is to segment a single Brief Motivational Intervention (BMI/BAI)
session by identifying only the START segment_id of each session component that
is PRESENT in the transcript.

The possible session components are listed below. NOT ALL COMPONENTS WILL
APPEAR IN EVERY SESSION. Only label a component if you can identify a specific
opening segment that clearly matches that component's canonical opening pattern.

===========================================================================
SESSION COMPONENT LIST (in typical order, but order may vary between sessions)
===========================================================================

1. Introduction
2. Decisional Balance Exercise
3. Feedback: Drinking Data and BAC
4. Feedback: Consequences and Risks
5. Feedback Summary
6. Envision the Future
7. Establish Goals
8. Strategies for Reducing Drinking/Drug Use
9. Conclusion of Session

===========================================================================
CRITICAL RULES
===========================================================================

1. ALL components are OPTIONAL, including Introduction.
   - Do NOT predict a component simply to fill the schema.
   - If a component is absent from this transcript, omit it from the output
     entirely.
   - If Introduction is present, it will almost always start at segment_id 0;
     however, if the first segment does not match an Introduction opening
     pattern, do not force it.

2. A component boundary is valid ONLY when the opening segment clearly matches
   that component's CANONICAL OPENING PATTERN.
   Vague topical relevance is NOT sufficient. You must be able to point to a
   specific sentence or phrase that matches a listed pattern.

3. Components DO NOT overlap. Each segment belongs to exactly one component.
   The end of one component is implicitly the segment immediately before the
   next component begins.

4. Brief references or digressions do NOT constitute a component transition.
   The conversational focus must clearly and sustainedly shift.

5. start_indices must be strictly increasing.

===========================================================================
SESSION COMPONENT DEFINITIONS AND CANONICAL OPENING PATTERNS
===========================================================================

[Introduction]
Function: Greeting, rapport building, explaining session purpose, describing
  confidentiality or session structure, and transitioning into the intervention.
Opening patterns:
  - "Hi, thanks for coming in today."
  - "Good to meet you. My name is X and today we're going to..."
  - "Today we're going to talk about your drinking / drug use."
  - "Before we get started, I want to explain how today will work."
  - "This is a private session; anything you say here is confidential unless..."
  - "We have about 45 minutes together today."
  - "So I understand you filled out some questionnaires last week..."
  - "The purpose of today's session is to go over some information about your alcohol use."
  - Any warm greeting or session framing at the very beginning of the recording.

[Decisional Balance Exercise]
Function: Explicitly eliciting the student's perceived pros and cons of drinking
  or drug use, exploring ambivalence, WITHOUT showing printed assessment data.
Opening patterns:
  - "I'd like to start by asking you what you like about drinking."
  - "What are some good things about drinking for you? What do you enjoy?"
  - "What do you get out of drinking / using?"
  - "Can you tell me what you like about alcohol?"
  - "Let's talk about the pros and cons of drinking for you."
  - "So what's the upside of drinking for you?"
  - "What are some of the positive things you get out of it?"
  - "And what are some things you don't like as much about drinking?"
  - "What are the downsides or things you're less happy about?"
  - "On the one hand you've mentioned X... on the other hand..."
  - "So it sounds like drinking gives you [benefit]; what about things that are
    less great about it?"

[Feedback: Drinking Data and BAC]
Function: Presenting quantitative, data-driven feedback from the assessment:
  the student's own drinking quantity and frequency compared to college-student
  norms, PLUS explanation of BAC and the student's personal BAC estimates on
  two nights with charts. Everything in this component is about showing and
  interpreting NUMBERS.
Opening patterns:
  - "Now I'd like to go over some of the information from the questionnaire you filled out."
  - "So we have your feedback form here, let me walk you through it."
  - "This first part of the feedback shows how much you drink compared to other students."
  - "According to what you reported, you drink about X drinks per week."
  - "Compared to other college students your age, your drinking is at about the Xth percentile."
  - "Looking at this table, you can see where your drinking falls relative to your peers."
  - "So about X% of students your age drink less than you do."
  - "The next section here is about Blood Alcohol Concentration, or BAC."
  - "BAC, Blood Alcohol Concentration, is a measure of the amount of alcohol in your bloodstream."
  - "I want to explain what BAC is and what affects it."
  - "Now let's look at your feedback form and see where your BACs fall."
  - "Based on what you told us, on one of your lighter nights your BAC was about X."
  - "We printed graphs to show your BAC over the course of two drinking nights."
  - "On your heavier night you had X drinks in X hours, giving you a peak BAC of X."
  - "This is the legal driving limit [pointing at graph]... your BAC was above that for X hours."

[Feedback: Consequences and Risks]
Function: Reviewing the negative consequences the student personally endorsed,
  health and sexual risks, alcohol-related problems in specific life domains,
  dependence risk factors (if applicable), drug use patterns (if applicable),
  and the financial/caloric costs of drinking. The defining shift from the
  previous component is moving from "here are your numbers" to "here are the
  things that have happened to you / risks you face."
Opening patterns:
  - "This next part of the feedback looks at some of the consequences associated with your drinking."
  - "On this next page we have a list of consequences that students sometimes experience."
  - "You indicated on the questionnaire that you have experienced [blackouts / injuries / ...]"
  - "I want to talk about some of the risks that go along with drinking at this level."
  - "So looking at this consequences section..."
  - "Let's talk about some of the things you've noticed happening as a result of your drinking."
  - "You mentioned some negative things that have happened, let's look at those more closely."
  - "The next section looks at your risk for alcohol dependence."
  - "Alcohol dependence is a condition that develops after several years of heavy drinking..."
  - "We also summarize your drug / marijuana use in this section."
  - "You reported using marijuana about X times per week, let me summarize that here."
  - "This next section looks at the money you spend on alcohol and drugs."
  - "You spend about $X a month on alcohol, which adds up to $X over the year."
  - "This section looks at the calories you consume from alcohol."
  - "Each standard drink has about 100 to 150 calories; on your heavier nights that's X calories."

[Feedback Summary]
Function: A global wrap-up of ALL the feedback material; the therapist
  acknowledges that it was a lot of information and invites an overall reaction
  before moving on.
Opening patterns:
  - "OK, so I appreciate your patience going through all of that, I know it was a lot."
  - "So that's everything we have on the feedback form."
  - "To summarize everything we just looked at..."
  - "Let me pull together the main points from the feedback before we move on."
  - "What's your overall reaction to all of that information?"
  - "What do you make of everything we just went through?"
  - "What part of that stood out to you the most?"
  - "What surprised you most from all of this?"
  - "I'm curious what you think about all of that overall."
  - "So that is all of the feedback, what questions do you have?"

[Envision the Future]
Function: Asking the student to project forward, imagining future consequences
  of continued drinking OR the benefits of change, as a motivational bridge
  before explicit goal-setting.
Opening patterns:
  - "What do you think will happen if you continue to drink the same way you have been?"
  - "Where do you see yourself in a year or two if things stay the same?"
  - "What do you think would happen if you cut down on your drinking?"
  - "What would be some advantages of cutting back?"
  - "Can you imagine what your life might look like if you drank a little less?"
  - "How do you think your [grades / relationships / health] might change if you drank less?"
  - "I'd like you to think about where your drinking might take you in the future."
  - "What are you most worried about if things don't change?"

[Establish Goals]
Function: Explicitly establishing the student's intention or plan for changing
  their drinking or drug use; exploring commitment and next steps.
Opening patterns:
  - "Where does this leave you now?"
  - "What would you like to do about your drinking / drug use?"
  - "Is cutting back something you're interested in?"
  - "What do you think your next step might be?"
  - "How do you see yourself making use of all this information?"
  - "What would you like things to look like differently?"
  - "What changes, if any, are you thinking about making?"
  - "Let's talk about whether or not you'd like to set a goal around your drinking."
  - "Would you be interested in thinking about a specific goal?"
  - "So you've said you'd like to cut back, what does that look like for you?"
  - "What do you think has to change?"

[Strategies for Reducing Drinking/Drug Use]
Function: Providing or co-generating concrete harm-reduction strategies, tips,
  and action plans for safer drinking or drug-use reduction.
Opening patterns:
  - "Would it be OK if I went over some strategies for reducing your drinking?"
  - "I'd like to share some strategies that other students have found helpful."
  - "We have a list of strategies here, let me walk you through them."
  - "Here are some tips for safer drinking."
  - "Some students find it helpful to space drinks out, alternate with water, or set a limit."
  - "What ideas do you have for keeping the positive aspects of drinking while avoiding the downsides?"
  - "Let's brainstorm some ways you might be able to stick to your goal."
  - "What strategies have worked for you in the past?"
  - "One approach a lot of people try is to count drinks ahead of time and set a hard limit."
  - "Have you ever tried alternating alcoholic and non-alcoholic drinks?"
  - "What might get in your way when trying to stick to your goal?"

[Conclusion of Session]
Function: Final global summary of the whole session, affirming the student's
  autonomy and effort, and bringing the interaction to a close.
Opening patterns:
  - "Let me summarize what we talked about today."
  - "So we covered a lot today, we talked about [topics]."
  - "What you do with all of this information is completely up to you."
  - "Thanks for being so open with me today, I really appreciate it."
  - "That honesty takes real courage, and I want you to know that."
  - "I want to wrap up by reviewing the main things we discussed."
  - "We're just about out of time, let me give a quick summary."
  - "Before you leave, I want to summarize the key points."
  - "That's the end of our session today."
  - "Do you have any final questions before we wrap up?"
  - "It was great talking with you, best of luck with your goals."

===========================================================================
INPUT
===========================================================================

You will be given Step 3 coded segments in chronological order.
Each item has the form:
<segment_id>: <segment text>

===========================================================================
YOUR TASK
===========================================================================

1. Boundary analysis:
   - Scan the transcript and identify segments whose text matches a CANONICAL
     OPENING PATTERN for a component.
   - For each candidate boundary, quote the specific phrase that triggers it
     and name which canonical pattern it matches.
   - If no segment clearly matches a component's canonical opening, DO NOT
     include that component in the output.
   - Verify that all recorded start_indices are strictly increasing.
   - You do NOT need to determine end indices; they are derived automatically.

2. Final mapping:
   - Output ONLY the components that are actually present.
   - For each present component, report the start_index using the exact
     segment_id and the exact start segment text.

===========================================================================
OUTPUT FORMAT
===========================================================================

### Reasoning Process
[Step-by-step component boundary analysis, citing specific phrases.]

### Final JSON
```json
{
  "Introduction": {"start_index": 0, "start_utterance": "..."},
  "Decisional Balance Exercise": {"start_index": 12, "start_utterance": "..."},
  "Feedback: Drinking Data and BAC": {"start_index": 24, "start_utterance": "..."}
}
```
IMPORTANT: Omit any component key that is NOT present in this transcript.
Do NOT output a component with a null or placeholder value.
"""

SFAS_SEGMENT_PROMPT = r"""
You are an expert analyst of Substance-Free Activity Sessions (SFAS).

Your task is to segment a single SFAS session by identifying the START
segment_id of each session component that is PRESENT in the transcript.

The possible components are listed below. NOT ALL COMPONENTS WILL APPEAR IN
EVERY SESSION, and SOME COMPONENTS MAY APPEAR MULTIPLE TIMES. Label a component
boundary whenever you can identify a segment that clearly matches that
component's canonical opening pattern.

===========================================================================
SESSION COMPONENT LIST
===========================================================================

1. Introduction
2. College/Career Goals
3. Alcohol/Drug Impact
4. Graduation Benefits
5. Career Requirements
6. Career Activities
7. Time Allocation
8. Substance-Free Activities
9. Summary/Goal Setting

===========================================================================
CRITICAL RULES
===========================================================================

1. The conversation can be NON-LINEAR. Components can appear OUT OF ORDER and
   MULTIPLE TIMES. Record every time the conversation clearly and sustainedly
   shifts to a new component.
2. ALL components are OPTIONAL. If a component does not appear, do not include it.
3. A component boundary is valid ONLY when the opening segment clearly matches
   that component's CANONICAL OPENING PATTERN. Vague topical relevance is NOT
   sufficient.
4. Brief references or digressions do NOT constitute a component transition.
   The conversational focus must clearly and sustainedly shift.
5. The start_indices in the output list MUST be strictly increasing
   chronologically.

===========================================================================
SESSION COMPONENT DEFINITIONS AND CANONICAL OPENING PATTERNS
===========================================================================

[Introduction]
Function: Establishing rapport, following up regarding alcohol/drug use issues
  from the previous session, and explaining the purpose and format of the SFAS
  session.
Opening patterns:
  - "I enjoyed our discussion last week and wanted to see what questions or thoughts you had about our discussion or the feedback we provided?"
  - "The goal of our meeting today is to help you to get the most out of your college experience."
  - "Hopefully this session will help you to clarify your goals..."

[College/Career Goals]
Function: Discussing why the student decided to attend college, how college is
  going so far, and their goals for college and beyond.
Opening patterns:
  - "I'm curious to hear about some of the reasons why you decided to go to college?"
  - "What has college life been like so far?"
  - "What made you choose X as your major?"
  - "What are your goals for college? What would you like to accomplish..."

[Alcohol/Drug Impact]
Function: Exploring the extent to which the student believes that their
  alcohol/drug use has impinged on their goals or might impinge on them.
Opening patterns:
  - "Next I wanted to get your thoughts on the relationship between drinking and your goals for college and your career."
  - "How does your alcohol use fit in with your ability to accomplish your college goals?"
  - "Are there other ways in which alcohol impacts your academic performance?"

[Graduation Benefits]
Function: Providing didactic information about income differences between high
  school and college graduates, graduation rates, and the relationship between
  GPA and income.
Opening patterns:
  - "Here is some information on the financial impact of a college degree..."
  - "The average income for college grads is almost $70,000..."
  - "But, it might surprise you to hear that less than half of students who begin college ever graduate..."
  - "This next figure looks at the relationship between the grades you earn in college and your income."

[Career Requirements]
Function: Discussing the specific requirements for the student's desired career
  (or providing general information if undecided), such as GPA, internships,
  and graduate-school tests.
Opening patterns:
  - "I'd appreciate it if you could tell me what kind of career you would like to have."
  - "What are you doing to obtain your goals regarding [CAREER]?"
  - "We put together some information on [PROFESSION]. Would you like to go over this information with me?"
  - "I know that you haven't chosen a major yet... What we did was prepare some information for you about what other college students have found helpful when choosing a major."

[Career Activities]
Function: Providing information regarding specific campus and community
  activities (extracurriculars, clubs, volunteer work) that are consistent with
  the student's major or career goals.
Opening patterns:
  - "We put together some information on campus and community activities that are related to your major and intended career."
  - "The X club is for X majors, and would be a great way to meet professors..."
  - "Looking over all of these opportunities, if you were to pursue one, what would it be?"

[Time Allocation]
Function: Discussing the student's personal time allocation chart (hours spent
  on family, studying, drinking/drugs, etc.), the relationship between time
  allocation and GPA, and weighing current vs. future benefits of activities.
Opening patterns:
  - "If you're interested, I'd like to talk a bit about how you spend your time."
  - "I put together this chart of your average weekly time allocation based on the calendar measure you completed last week."
  - "To what extent is this graph consistent with your priorities for college?"
  - "Here is a graph that shows how GPA is related to time spent in various activities..."
  - "One thing that can be helpful when deciding what to do every day is to think about the benefits or value of various activities."

[Substance-Free Activities]
Function: Discussing non-academic, substance-free activities and hobbies the
  student enjoys, and providing information about recreational opportunities on
  campus or in the community.
Opening patterns:
  - "During the assessment you completed a few weeks ago, you provided a list of some activities that you said you enjoyed. We asked specifically about activities that did not involve drinking or using drugs."
  - "Which of these activities/hobbies do you enjoy most?"
  - "To help you learn about some of these substance-free activities, I have put together some information on how you can get involved..."

[Summary/Goal Setting]
Function: Summarizing the session, establishing specific short-term goals,
  completing the goal sheet, reviewing drinking goals, and offering handouts.
Opening patterns:
  - "We've talked about a number of ways that you can get the most out of your experience..."
  - "I'm curious about what stands out most about the information we covered?"
  - "As the final part of the session, I'd like to work with you on a goal setting exercise."
  - "What are some things that you can do this semester or this year to make progress towards those goals?"
  - "Finally, if you are interested there are two things I'd like to go over. First, if you are interested you can take this free day-planner."

===========================================================================
INPUT
===========================================================================

You will be given Step 3 coded segments in chronological order.
Each item has the form:
<segment_id>: <segment text>

===========================================================================
YOUR TASK
===========================================================================

1. Boundary analysis:
   - Scan the transcript and identify segments whose text matches a CANONICAL
     OPENING PATTERN for a component.
   - For each candidate boundary, quote the specific phrase that triggers it
     and name which canonical pattern it matches.
   - Record multiple boundaries if components recur or happen out of normal
     order.

2. Final mapping:
   - Output an ordered JSON LIST of component boundaries as they occur in time.
   - Use exact segment_id values as start_index values.

===========================================================================
OUTPUT FORMAT
===========================================================================

### Reasoning Process
[Step-by-step component boundary analysis, citing specific phrases.]

### Final JSON
```json
[
  {"section": "Introduction", "start_index": 0, "start_utterance": "..."},
  {"section": "Career Requirements", "start_index": 12, "start_utterance": "..."},
  {"section": "College/Career Goals", "start_index": 35, "start_utterance": "..."},
  {"section": "Career Requirements", "start_index": 68, "start_utterance": "..."}
]
```
IMPORTANT: The output MUST be a JSON array, starting with '[' and ending with ']'.
"""

SESSION_ALIASES = {
    "bmi": "bmi",
    "bai": "bmi",
    "blue2": "bmi",
    "blue-2": "bmi",
    "blue3": "bmi",
    "blue-3": "bmi",
    "sfas": "sfas",
    "safs": "sfas",
}

STEP1_TYPE_ROOTS = [
    ("sfas", "step1_output/SAFS_output"),
    ("bmi", "step1_output/BMI_output"),
    ("bmi", "step1_output/BAI"),
]


def normalize_session_type(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    if key == "auto":
        return "auto"
    if key not in SESSION_ALIASES:
        raise ValueError(f"Unknown session type: {value}")
    return SESSION_ALIASES[key]


def infer_session_type(path: str) -> str:
    lowered = path.lower()
    parts = [part for part in lowered.replace("\\", "/").split("/") if part]
    joined = " ".join(parts)
    if any(token in joined for token in ("sfas", "safs")):
        return "sfas"
    if any(token in joined for token in ("bmi", "bai", "blue-2", "blue2", "blue-3", "blue3")):
        return "bmi"
    sample_id = os.path.basename(path)
    for suffix in (".component_segmentation.csv", ".coded_segments.csv", ".segments.csv"):
        if sample_id.endswith(suffix):
            sample_id = sample_id[: -len(suffix)]
    if sample_id in ("", os.path.basename(path)):
        sample_id = os.path.basename(os.path.dirname(path))
    for session_type, root in STEP1_TYPE_ROOTS:
        if os.path.isdir(os.path.join(root, sample_id)):
            return session_type
    print(f"[WARN] Cannot infer session type from path; using BMI/BAI schema: {path}")
    return "bmi"


def config_for_session_type(session_type: str) -> Dict[str, str]:
    if session_type == "sfas":
        return {"prompt": SFAS_SEGMENT_PROMPT, "json_kind": "list"}
    if session_type == "bmi":
        return {"prompt": BMI_SEGMENT_PROMPT, "json_kind": "dict"}
    raise ValueError(f"Unsupported normalized session type: {session_type}")


def build_transcript(rows: List[Tuple[int, str]]) -> str:
    return "\n".join(f"{idx}: {text}" for idx, text in rows)


def extract_json(text: str, json_kind: str):
    final_json_match = re.search(r"###\s*Final JSON\s*\n", text, re.IGNORECASE)
    search_text = text[final_json_match.end():] if final_json_match else text
    search_text = re.sub(r"^```(?:json)?\s*", "", search_text.lstrip())
    search_text = re.sub(r"\s*```\s*$", "", search_text)

    if json_kind == "list":
        start, end = search_text.find("["), search_text.rfind("]")
    else:
        start, end = search_text.find("{"), search_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(search_text[start:end + 1])
    except Exception:
        return None


def filter_files(files: List[Tuple[str, str]], include: Optional[List[str]], exclude: Optional[List[str]]) -> List[Tuple[str, str]]:
    if include:
        files = [
            item for item in files
            if any(fnmatch.fnmatch(item[1], pat) or pat in item[1] for pat in include)
        ]
    if exclude:
        files = [
            item for item in files
            if not any(fnmatch.fnmatch(item[1], pat) or pat in item[1] for pat in exclude)
        ]
    return files


def collect_csv_files(root_dirs: List[str], glob_pattern: str) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    for root_dir in root_dirs:
        pattern = os.path.join(root_dir, glob_pattern)
        for path in sorted(glob.glob(pattern, recursive=True)):
            files.append((root_dir, path))
    return sorted(files, key=lambda item: item[1])


def output_path_for_csv(csv_path: str, root_dir: str, out_dir_mode: str, out_dir: str) -> str:
    base = os.path.basename(csv_path).replace(".coded_segments.csv", "").replace(".segments.csv", "")
    filename = f"{base}.component_segmentation.csv"
    if out_dir_mode == "per_folder":
        return os.path.join(os.path.dirname(csv_path), filename)
    if out_dir_mode == "flat":
        return os.path.join(out_dir, filename)
    rel = os.path.relpath(os.path.dirname(csv_path), root_dir)
    return os.path.join(out_dir, rel, filename)


def clean_output_path_for_component_csv(component_csv_path: str, out_dir: str, clean_out_dir: str) -> str:
    rel = os.path.relpath(component_csv_path, out_dir)
    cleaned_name = os.path.basename(rel).replace(".component_segmentation.csv", ".cleaned_step4.csv")
    rel_parent = os.path.dirname(rel)
    return os.path.join(clean_out_dir, rel_parent, cleaned_name)


def boundaries_from_parsed(parsed) -> List[Dict[str, object]]:
    boundaries: List[Dict[str, object]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                start = int(item.get("start_index"))
            except Exception:
                continue
            boundaries.append({
                "start_index": start,
                "session_component": item.get("section", ""),
                "component_start_utterance": item.get("start_utterance", ""),
            })
    elif isinstance(parsed, dict):
        for section, item in parsed.items():
            if not isinstance(item, dict):
                continue
            try:
                start = int(item.get("start_index"))
            except Exception:
                continue
            boundaries.append({
                "start_index": start,
                "session_component": section,
                "component_start_utterance": item.get("start_utterance", ""),
            })
    return sorted(boundaries, key=lambda row: row["start_index"])


def component_for_index(index: int, boundaries: List[Dict[str, object]]) -> Dict[str, object]:
    current: Optional[Dict[str, object]] = None
    for boundary in boundaries:
        if int(boundary["start_index"]) <= index:
            current = boundary
        else:
            break
    if current is None:
        return {
            "session_component": "",
            "component_start_index": "",
            "component_start_utterance": "",
        }
    return {
        "session_component": current.get("session_component", ""),
        "component_start_index": current.get("start_index", ""),
        "component_start_utterance": current.get("component_start_utterance", ""),
    }


def read_source_csv(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_coded_segments_csv(path: str) -> List[Tuple[int, str]]:
    """
    Return [(segment_id, text), ...] sorted by segment_id.
    The component model uses these Step 3 segment ids as start_index values.
    """
    rows: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "segment_id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns segment_id and text. Got: {reader.fieldnames}")
        for r in reader:
            try:
                segment_id = int(r["segment_id"])
            except Exception:
                continue
            text = r.get("text") or ""
            if text.strip() == "":
                continue
            rows.append((segment_id, text))
    rows.sort(key=lambda x: x[0])
    return rows


def write_component_csv(
    path: str,
    source_rows: List[Dict[str, object]],
    session_type: str,
    model: str,
    parsed_ok: bool,
    parsed,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    boundaries = boundaries_from_parsed(parsed)
    base_fields = list(source_rows[0].keys()) if source_rows else ["segment_id", "role", "code", "behavior", "motivation", "time", "text"]
    extra_fields = [
        "session_type", "model", "parsed_ok", "session_component",
        "component_start_index", "component_start_utterance",
    ]
    fieldnames = base_fields + [field for field in extra_fields if field not in base_fields]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in source_rows:
            try:
                segment_id = int(row.get("segment_id", ""))
            except Exception:
                segment_id = -1
            out = dict(row)
            out.update({
                "session_type": session_type,
                "model": model,
                "parsed_ok": parsed_ok,
            })
            out.update(component_for_index(segment_id, boundaries))
            writer.writerow(out)
    os.replace(tmp, path)


def write_clean_component_csv_from_rows(
    path: str,
    source_rows: List[Dict[str, object]],
    parsed,
    fields: List[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    boundaries = boundaries_from_parsed(parsed)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in source_rows:
            try:
                segment_id = int(row.get("segment_id", ""))
            except Exception:
                segment_id = -1
            out = dict(row)
            out.update(component_for_index(segment_id, boundaries))
            writer.writerow({field: out.get(field, "") for field in fields})
    os.replace(tmp, path)


def write_clean_component_csv(component_csv_path: str, clean_csv_path: str, fields: List[str]) -> None:
    os.makedirs(os.path.dirname(clean_csv_path), exist_ok=True)
    tmp = clean_csv_path + ".tmp"
    with open(component_csv_path, "r", encoding="utf-8", newline="") as in_f:
        reader = csv.DictReader(in_f)
        missing = [field for field in fields if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Cannot write cleaned CSV; missing columns {missing} in {component_csv_path}")
        with open(tmp, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, clean_csv_path)


async def call_chat(
    client: AsyncOpenAI,
    model: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert annotation assistant. "
                    "Follow the output format exactly as specified in the user prompt: "
                    "first output your reasoning under '### Reasoning Process', "
                    "then output the JSON under '### Final JSON'."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


async def main_async(args):
    api_key = args.api_key or "token-abc123"
    base_url = args.openai_api_base or "http://localhost:8000/v1"
    session_override = normalize_session_type(args.session_type)

    csv_files = collect_csv_files(args.root_dir, args.glob)
    csv_files = filter_files(csv_files, args.include, args.exclude)
    if args.range:
        start_idx, end_idx = args.range
        csv_files = csv_files[start_idx:end_idx]
    if args.max_files > 0:
        csv_files = csv_files[:args.max_files]
    if not csv_files:
        raise RuntimeError("No CSV files remain after filtering.")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print(f"[INFO] Found {len(csv_files)} csv files after filtering.")
    print(f"[INFO] base_url: {base_url}")
    print(f"[INFO] models: {args.models}")
    print(f"[INFO] session_type: {args.session_type}")

    for root_dir, csv_path in csv_files:
        session_type = infer_session_type(csv_path) if session_override == "auto" else session_override
        cfg = config_for_session_type(session_type)

        try:
            rows = read_coded_segments_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Failed reading {csv_path}: {exc}")
            continue

        transcript = build_transcript(rows)
        user_content = (
            cfg["prompt"]
            + "\n\nThe numbered items below are Step 3 coded segments. "
            "Use these segment_id numbers as start_index values in the final JSON."
            + "\n\n--- Coded Segments ---\n"
            + transcript
        )

        full_out_path = output_path_for_csv(csv_path, root_dir, args.out_dir_mode, args.out_dir)
        clean_out_path = clean_output_path_for_component_csv(full_out_path, args.out_dir, args.clean_out_dir)
        if args.skip_existing and os.path.exists(clean_out_path):
            print(f"[SKIP] {csv_path} (already exists)")
            continue
        source_rows = read_source_csv(csv_path)

        print(f"\n[FILE] {csv_path}")
        print(f"[TYPE] {session_type}")
        if args.write_full_output:
            print(f"[FULL] {full_out_path}")
        print(f"[OUT ] {clean_out_path}")

        for model in args.models:
            t0 = time.time()
            try:
                content = await call_chat(
                    client=client,
                    model=model,
                    user_content=user_content,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                parsed = extract_json(content, cfg["json_kind"])
                parsed_ok = parsed is not None
                if args.write_full_output:
                    write_component_csv(full_out_path, source_rows, session_type, model, parsed_ok, parsed)
                write_clean_component_csv_from_rows(clean_out_path, source_rows, parsed, DEFAULT_CLEAN_FIELDS)
                print(f"[DONE] {model} ({round(time.time() - t0, 3)}s) parsed_ok={parsed_ok}")
            except Exception as exc:
                if args.write_full_output:
                    write_component_csv(full_out_path, source_rows, session_type, model, False, None)
                write_clean_component_csv_from_rows(clean_out_path, source_rows, None, DEFAULT_CLEAN_FIELDS)
                print(f"[ERR ] {model}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified BMI/BAI/SFAS session-component segmentation runner")
    parser.add_argument(
        "--root_dir",
        nargs="+",
        default=[
            "speaker_merged_turns",
        ],
        help="One or more directories containing Step 3 *.coded_segments.csv files",
    )
    parser.add_argument("--glob", default="**/*.coded_segments.csv")
    parser.add_argument(
        "--session_type",
        choices=["auto", "bmi", "bai", "sfas", "safs"],
        default="auto",
        help="auto infers per file from path; bai uses the BMI prompt",
    )
    parser.add_argument(
        "--out_dir_mode",
        choices=["per_folder", "central", "flat"],
        default="central",
        help="per_folder writes next to CSV; flat writes all files in out_dir; central preserves relative structure",
    )
    parser.add_argument("--out_dir", default="component_segments")
    parser.add_argument("--clean_out_dir", default="cleaned-final")
    parser.add_argument("--write_full_output", action="store_true", help="Also write full Step 4 CSVs under --out_dir")
    parser.add_argument("--models", nargs="+", required=True, help="One or more model ids supported by your vLLM server")
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--api_key", default="", help="vLLM serve --api-key")
    parser.add_argument("--openai_api_base", default="", help="e.g., http://localhost:8000/v1")
    parser.add_argument("--include", nargs="+", default=None, metavar="PATTERN")
    parser.add_argument("--exclude", nargs="+", default=None, metavar="PATTERN")
    parser.add_argument("--range", nargs=2, type=int, default=None, metavar=("START", "END"))
    parser.add_argument("--max_files", type=int, default=0, help="Maximum total files to process; 0 means no limit")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
