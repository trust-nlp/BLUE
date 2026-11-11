#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified session-component segmentation runner for AAAA-BLUE2+3.

For each Step 3 *.coded_segments.csv file, this program chooses the BMI, BAI or
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


UNKNOWN_COMPONENT = "Unknown"

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

Your task is to segment a single Brief Motivational Intervention (BMI)
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
   When the therapist announces the move ("we can jump into the feedback part"),
   the component starts at that announcing sentence, not at the first content.
   This also holds when the announcement comes at the END of a turn that first
   wraps up the previous topic ("we'll come back to this later... if it's okay,
   let's go through the feedback", "so that's it for the feedback, let's jump back
   into goal setting"): the new component starts at THAT turn, not at the turn
   after the client's "Okay" where the content begins.

4. Brief references or digressions do NOT constitute a component transition.
   The conversational focus must clearly and sustainedly shift.

5. start_indices must be strictly increasing. A component may appear more than
   once when the focus clearly returns to it later, for example remaining feedback
   pages after a goal discussion. Record each return as a new entry.

6. The recording may be INCOMPLETE. It may start in the middle of the session
   (e.g. the first segment is a half sentence, or the session is already in
   the feedback) and it may end before the session is over.
   - Do not assume segment_id 0 is the Introduction. But if the first segments
     frame the session (its purpose, what will be covered, that decisions are up
     to the student), label them Introduction even if the greeting was not recorded.
   - If the transcript starts in the middle of a component whose opening was
     not recorded, and the content of the first segments clearly belongs to
     that component, you may label that component starting at the first
     segment_id even though its canonical opening is missing.
   - If you cannot tell which component the first segments belong to, leave
     them unlabeled (start your first component later).
   - Do not invent a Conclusion of Session if the recording is cut off before it.

7. The optional modules (Biphasic Effect, Expectancies) can be administered at
   any point. They do NOT start a new component; they belong to whichever
   component is in progress.

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
  Saying that students find it helpful to maximize the good parts of drinking is
  still Introduction; this component starts when the therapist asks for the
  student's own thoughts.
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
Function: The risk factors section of the feedback, which opens with heavy
  drinking: the heavy/binge-drinking cut-off (5+ drinks per occasion, 4+ for
  women) and the student's number of heavy drinking days. Also reviewing the
  negative consequences the student personally endorsed, health and sexual risks, alcohol-related problems in specific life domains,
  dependence risk factors (if applicable), drug use patterns (if applicable),
  and the financial/caloric costs of drinking. The defining shift from the
  previous component is moving from "here are your numbers" to "here are the
  things that have happened to you / risks you face."
Opening patterns:
  - "The next sections address some of the risk factors associated with your drinking pattern."
  - "The first part looks at heavy drinking, which is defined as consuming 5 or more drinks during one drinking occasion."
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
  their drinking or drug use; exploring commitment and next steps. Therapists
  often do this without a listed sentence, so no exact wording is required: it
  starts when the therapist asks whether, when or how the student would change
  and the talk stays there. A listed sentence such as "How do you see yourself
  making use of all this information?" starts it even when the student answers
  briefly. An unrelated single question in passing does not count.
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
  - "At what point would you feel like you would want to cut back?"
  - "What would have to happen for you to want to change?"

[Strategies for Reducing Drinking/Drug Use]
Function: Providing or co-generating concrete harm-reduction strategies, tips,
  and action plans for safer drinking or drug-use reduction. It starts when the
  therapist moves to strategies and the talk stays there, by asking for the
  student's ideas or presenting the list. Only mentioning that a handout exists
  does not start it.
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
  autonomy and effort, and bringing the interaction to a close. It often follows
  Strategies with no explicit announcement: once the therapist stops discussing
  individual strategies and recaps the session as a whole, Conclusion of
  Session has begun. It starts at the recap when there is one, even if the thanks
  come later. A closing statement can start it when there is no recap. Only asking
  "any questions?" is not a Conclusion of Session.
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
  - "It was great talking with you, best of luck with your goals."
  - "It sounds like a lot from this handout / today could be helpful for you..."
  - "So it sounds like what stood out to you was X, and you're going to Y."

===========================================================================
INPUT
===========================================================================

You will be given Step 3 coded segments in chronological order.
Each item has the form:
<segment_id>: [therapist|patient] <segment text>

The role tag shows who is speaking. Component openings are almost always
spoken by the therapist; patient segments show how the client responds.

===========================================================================
YOUR TASK
===========================================================================

1. Boundary analysis:
   - Scan the transcript and identify segments whose text matches a CANONICAL
     OPENING PATTERN for a component.
   - For each candidate boundary, quote the specific phrase that triggers it
     and name which canonical pattern it matches.
   - If no segment clearly matches a component's canonical opening, DO NOT
     include that component in the output (except for the incomplete-recording
     case in rule 6).
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
[
  {"section": "Introduction", "start_index": 0, "start_utterance": "..."},
  {"section": "Decisional Balance Exercise", "start_index": 12, "start_utterance": "..."},
  {"section": "Feedback: Drinking Data and BAC", "start_index": 24, "start_utterance": "..."}
]
```
IMPORTANT: The output MUST be a JSON array, starting with '[' and ending with ']'.
Omit any component that is NOT present in this transcript.
"""

BAI_SEGMENT_PROMPT = r"""
You are an expert analyst of Motivational Interviewing (MI) sessions.

Your task is to segment a single BAI motivational intervention session by
identifying only the START segment_id of each session component that is PRESENT
in the transcript.

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
5. Feedback Summary & Establish Goals
6. Strategies for Reducing Drinking/Drug Use
7. Conclusion of Session

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
   When the therapist announces the move ("we can jump into the feedback part"),
   the component starts at that announcing sentence, not at the first content.
   This also holds when the announcement comes at the END of a turn that first
   wraps up the previous topic ("we'll come back to this later... if it's okay,
   let's go through the feedback", "so that's it for the feedback, let's jump back
   into goal setting"): the new component starts at THAT turn, not at the turn
   after the client's "Okay" where the content begins.

4. Brief references or digressions do NOT constitute a component transition.
   The conversational focus must clearly and sustainedly shift.

5. start_indices must be strictly increasing. A component may appear more than
   once when the focus clearly returns to it later, for example remaining feedback
   pages after a goal discussion. Record each return as a new entry.

6. The recording may be INCOMPLETE. It may start in the middle of the session
   (e.g. the first segment is a half sentence, or the session is already in
   the feedback) and it may end before the session is over.
   - Do not assume segment_id 0 is the Introduction. But if the first segments
     frame the session (its purpose, what will be covered, that decisions are up
     to the student), label them Introduction even if the greeting was not recorded.
   - If the transcript starts in the middle of a component whose opening was
     not recorded, and the content of the first segments clearly belongs to
     that component, you may label that component starting at the first
     segment_id even though its canonical opening is missing.
   - If you cannot tell which component the first segments belong to, leave
     them unlabeled (start your first component later).
   - Do not invent a Conclusion of Session if the recording is cut off before it.

7. The optional handouts (Biphasic Effect, Expectancies, Driving After Using
   Marijuana) can be used at any point. They do NOT start a new component; they
   belong to whichever component is in progress.

===========================================================================
SESSION COMPONENT DEFINITIONS AND CANONICAL OPENING PATTERNS
===========================================================================

[Introduction]
Function: Greeting, rapport building, explaining session purpose, and describing
  the feedback, confidentiality or session structure.
Opening patterns:
  - "Thanks again for taking part in this project."
  - "The purpose of this meeting today is to spend some time talking about your experiences with alcohol."
  - "We put together a feedback sheet based on that information."
  - "Today we're going to talk about your drinking."
  - Any warm greeting or session framing at the very beginning of the recording.

[Decisional Balance Exercise]
Function: Eliciting what the participant likes and does not like about drinking
  or drug use, WITHOUT showing printed assessment data. Saying that people find it
  helpful to maximize the good parts of drinking is still Introduction; this
  component starts when the therapist asks for the participant's own thoughts.
Opening patterns:
  - "I wanted to start by asking about your general thoughts about drinking."
  - "What do you like about drinking?"
  - "What don't you like as much about drinking?"
  - "So it sounds like drinking gives you [benefit]; what about things that are
    less great about it?"

[Feedback: Drinking Data and BAC]
Function: Feedback Pages 1-3: the participant's beliefs about how much others
  drink, their drinking patterns compared to norms, BAC and personal BAC
  estimates, PLUS the "Moderate vs. Heavy Drinking" section: the heavy/binge
  drinking cut-off (5+ drinks per occasion, 4+ for females), its relation to
  health, and the recommended moderate drinking guidelines. The binge-drinking
  definition belongs HERE, before the Alcohol-Related Consequences section.
Opening patterns:
  - "As I mentioned before, I have prepared a personalized feedback form based on the information you provided in the survey."
  - "This first section looks at your estimate of how much other people drink."
  - "Compared to other people your age, your drinking is at about the Xth percentile."
  - "The next section here is about Blood Alcohol Concentration, or BAC."
  - "We computed your BAC for two of your drinking nights."
  - "This next section is about how many drinks you might have in a given day."
  - "The next section looks at heavy drinking, which is defined as consuming 5 or more drinks during one drinking occasion."
  - "Do you know what that moderate drinking amount is?"

[Feedback: Consequences and Risks]
Function: The Alcohol-Related Consequences section and everything after it on
  the feedback form: consequences and risk behaviors the participant reported,
  alcohol use disorder symptoms, interactions with medications and other drugs,
  and the money and calories spent on drinking.
Opening patterns:
  - "This next section looks at some of the possible alcohol-related consequences or risk behaviors you reported on the survey."
  - "I'm curious which of these outcomes stands out to you?"
  - "Which of these outcomes bother you the most or would you most like to avoid going forward?"
  - "These consequences are similar to symptoms of an alcohol use disorder."
  - "This next section looks at the effects of combining alcohol and other medications or drugs."
  - "This next section looks at the money you spend on alcohol and other drugs."
  - "This next section summarizes the amount of calories you consume from alcohol."

[Feedback Summary & Establish Goals]
Function: One combined section: inviting an overall reaction to all the feedback,
  establishing the participant's interest in changing, and establishing a plan.
  Do NOT start a new component when the conversation moves from the overall
  reaction to goals.
Opening patterns:
  - "I appreciate your patience as we reviewed all of this material."
  - "I'm curious to hear your overall reaction. What do you make of all of this?"
  - "What part of our discussion was most interesting to you?"
  - "Where does this leave you now?"
  - "How do you see yourself making use of all this information?"
  - "How would you like things to be different?"
  - "What do you think might be your next step?"

[Strategies for Reducing Drinking/Drug Use]
Function: Reviewing strategies for reducing drinking or drug use and related
  problems, setting a specific goal, and discussing barriers to change.
Opening patterns:
  - "Would you be interested in hearing about ways to avoid some of the negative consequences?"
  - "We have a list of strategies for reducing the risks associated with drinking."
  - "Which of these safer drinking strategies do you find to be most helpful or important?"
  - "Are there any new strategies that might be helpful for you to try going forward?"
  - "What might get in your way when trying to stick to a limit?"

[Conclusion of Session]
Function: A brief grand summary of the whole session (feedback, motivation, plan,
  self-efficacy), affirming the participant's autonomy and bringing the
  interaction to a close. It can be abbreviated if the session runs out of time.
  It starts at the summary when there is one, even if the thanks come later. Only
  asking "any questions?" is not a Conclusion of Session.
Opening patterns:
  - "I appreciate your openness and honesty in talking to me."
  - "Let me try to summarize all that we've covered today."
  - "Remember that what you do with the information discussed today is totally up to you."

===========================================================================
INPUT
===========================================================================

You will be given Step 3 coded segments in chronological order.
Each item has the form:
<segment_id>: [therapist|patient] <segment text>

The role tag shows who is speaking. Component openings are almost always
spoken by the therapist; patient segments show how the client responds.

===========================================================================
YOUR TASK
===========================================================================

1. Boundary analysis:
   - Scan the transcript and identify segments whose text matches a CANONICAL
     OPENING PATTERN for a component.
   - For each candidate boundary, quote the specific phrase that triggers it
     and name which canonical pattern it matches.
   - If no segment clearly matches a component's canonical opening, DO NOT
     include that component in the output (except for the incomplete-recording
     case in rule 6).
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
[
  {"section": "Introduction", "start_index": 0, "start_utterance": "..."},
  {"section": "Decisional Balance Exercise", "start_index": 12, "start_utterance": "..."},
  {"section": "Feedback: Drinking Data and BAC", "start_index": 24, "start_utterance": "..."}
]
```
IMPORTANT: The output MUST be a JSON array, starting with '[' and ending with ']'.
Omit any component that is NOT present in this transcript.
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
3. Graduation Benefits
4. Career Requirements
5. Career Activities
6. Time Allocation
7. Coping with Stress (optional module)
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
   The conversational focus must clearly and sustainedly shift. When the therapist
   announces the move ("the next page looks at..."), the component starts at that
   announcing sentence, not at the first content.
   This also holds when the announcement comes at the END of a turn that first
   wraps up the previous topic ("we'll come back to this later... if it's okay,
   let's go through the feedback", "so that's it for the feedback, let's jump back
   into goal setting"): the new component starts at THAT turn, not at the turn
   after the client's "Okay" where the content begins.
5. The start_indices in the output list MUST be strictly increasing
   chronologically.
6. The recording may be INCOMPLETE. It may start in the middle of the session
   (e.g. the first segment is a half sentence, or the session is already past
   the introduction) and it may end before the session is over.
   - Do not assume segment_id 0 is the Introduction. But if the first segments
     frame the session (its purpose, what will be covered, that decisions are up
     to the student), label them Introduction even if the greeting was not recorded.
   - If the transcript starts in the middle of a component whose opening was
     not recorded, and the content of the first segments clearly belongs to
     that component, you may label that component starting at the first
     segment_id even though its canonical opening is missing.
   - If you cannot tell which component the first segments belong to, leave
     them unlabeled (start your first component later).
   - Do not invent a Summary/Goal Setting if the recording is cut off before it.
7. The self-control handout can be used at any point. It does NOT start a new
   component; it belongs to whichever component is in progress.

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
  going so far, their goals for college and beyond, and the relationship between
  their alcohol/drug use and those goals. Talk about how alcohol or drugs fit with
  these goals stays in this component and does not start a new one.
  The therapist often starts the goal sheet here by recording the student's goals.
  This is not Summary/Goal Setting, which comes near the end of the session.
Opening patterns:
  - "I'm curious to hear about some of the reasons why you decided to go to college?"
  - "What has college life been like so far?"
  - "What made you choose X as your major?"
  - "What are your goals for college? What would you like to accomplish..."

[Graduation Benefits]
Function: Providing didactic information about income differences between high
  school and college graduates, graduation rates, and the relationship between
  GPA and income.
Opening patterns:
  - Graduation Benefits is the first feedback page, so the therapist's
    announcement of the feedback starts it: "If it's okay with you, we can jump
    into the feedback part", "I'd like to go through this feedback based on the
    questionnaire you filled out", "So here's the feedback stuff".
  - "Here is some information on the financial impact of a college degree..."
  - "The average income for college grads is almost $70,000..."
  - "But, it might surprise you to hear that less than half of students who begin college ever graduate..."
  - "This next figure looks at the relationship between the grades you earn in college and your income."

[Career Requirements]
Function: Discussing the specific requirements for the student's desired career
  (or providing general information if undecided), such as GPA, internships,
  and graduate-school tests. Briefly reading extracurricular activities or
  student organizations off the career handout remains Career Requirements.
Opening patterns:
  - "I'd appreciate it if you could tell me what kind of career you would like to have."
  - "What are you doing to obtain your goals regarding [CAREER]?"
  - "We put together some information on [PROFESSION]. Would you like to go over this information with me?"
  - "I know that you haven't chosen a major yet... What we did was prepare some information for you about what other college students have found helpful when choosing a major."

[Career Activities]
Function: Providing information regarding specific campus and community
  activities (extracurriculars, clubs, volunteer work) that are consistent with
  the student's major or career goals. It starts when the therapist turns to
  these activities and engages the student about them, such as asking about
  interest, fit or benefit, even when they appear on the same handout as the
  career requirements.
Opening patterns:
  - "We put together some information on campus and community activities that are related to your major and intended career."
  - "The X club is for X majors, and would be a great way to meet professors..."
  - "Looking over all of these opportunities, if you were to pursue one, what would it be?"
  - "This part shows you the extracurricular activities that are available on campus."
  - "What do you think of any of these as helping to fit with the goals you mentioned?"

[Time Allocation]
Function: Discussing the student's personal time allocation chart (hours spent
  on family, studying, drinking/drugs, etc.), the relationship between time
  allocation and GPA, and weighing current vs. future benefits of activities.
  The exercise of rating activities on their current and future value belongs here
  even when it comes after the goal sheet. When the therapist returns to the goal
  sheet afterwards, start a new Summary/Goal Setting entry.
Opening patterns:
  - "If you're interested, I'd like to talk a bit about how you spend your time."
  - "I put together this chart of your average weekly time allocation based on the calendar measure you completed last week."
  - "To what extent is this graph consistent with your priorities for college?"
  - "Here is a graph that shows how GPA is related to time spent in various activities..."
  - "One thing that can be helpful when deciding what to do every day is to think about the benefits or value of various activities."

[Coping with Stress]
Function: Optional module for students who report elevated stress, anxiety or
  depression: feedback on the student's mood and coping skills, the role of
  alcohol in stress, and suggestions for other ways of coping. It usually
  follows Time Allocation.
Opening patterns:
  - "The next thing that I'd like to talk to you about is stress."
  - "College can be a very stressful time, and many students report having difficulties with their moods."
  - "These are some things that you told us about your mood. Does that sound about right?"
  - "Are there times when you feel more stressed out than others?"
  - "How does alcohol use relate to your stress level or mood?"

[Substance-Free Activities]
Function: Discussing non-academic, substance-free activities and hobbies the
  student enjoys, and providing information about recreational opportunities on
  campus or in the community. It often follows Coping with Stress with no
  announcement: it starts when the therapist presents the activities the student
  reported enjoying or other recreational options, rather than ways to cope.
Opening patterns:
  - "These are some activities that you report doing, and other activities that you may enjoy."
  - "During the assessment you completed a few weeks ago, you provided a list of some activities that you said you enjoyed. We asked specifically about activities that did not involve drinking or using drugs."
  - "Which of these activities/hobbies do you enjoy most?"
  - "To help you learn about some of these substance-free activities, I have put together some information on how you can get involved..."

[Summary/Goal Setting]
Function: Summarizing the session, establishing specific short-term goals,
  completing the goal sheet, reviewing drinking goals, and offering handouts.
Opening patterns:
  - The therapist closing the feedback and moving back to goals: "So that is
    really it for the feedback part", "So kind of wrapping up... then look at some
    goal setting", "Let's jump back into the goal setting".
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
<segment_id>: [therapist|patient] <segment text>

The role tag shows who is speaking. Component openings are almost always
spoken by the therapist; patient segments show how the client responds.

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
    "bai": "bai",
    "blue2": "bmi",
    "blue-2": "bmi",
    "blue3": "bai",
    "blue-3": "bai",
    "sfas": "sfas",
    "safs": "sfas",
}

STEP1_TYPE_ROOTS = [
    ("sfas", "step1_output/SAFS_output"),
    ("bmi", "step1_output/BMI_output"),
    ("bai", "step1_output/BAI"),
]


def normalize_session_type(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    if key == "auto":
        return "auto"
    if key not in SESSION_ALIASES:
        raise ValueError(f"Unknown session type: {value}")
    return SESSION_ALIASES[key]


def infer_session_type(path: str) -> str:
    sample_id = os.path.basename(path)
    for suffix in (".component_segmentation.csv", ".coded_segments.csv", ".segments.csv"):
        if sample_id.endswith(suffix):
            sample_id = sample_id[: -len(suffix)]
    if sample_id in ("", os.path.basename(path)):
        sample_id = os.path.basename(os.path.dirname(path))
    # Session names decide first: the project path itself contains "BLUE2".
    for token, session_type in (("sfas", "sfas"), ("safs", "sfas"), ("bai", "bai"), ("bmi", "bmi")):
        if re.search(rf"(?<![a-z]){token}(?![a-z])", sample_id.lower()):
            return session_type
    for session_type, root in STEP1_TYPE_ROOTS:
        if os.path.isdir(os.path.join(root, sample_id)):
            return session_type
    lowered = path.lower().replace("\\", "/")
    for token, session_type in (("/bai", "bai"), ("/blue-3", "bai"), ("/blue3", "bai"), ("/bmi", "bmi"), ("/blue-2", "bmi")):
        if token in lowered:
            return session_type
    print(f"[WARN] Cannot infer session type from path; using BMI schema: {path}")
    return "bmi"


def config_for_session_type(session_type: str) -> Dict[str, str]:
    if session_type == "sfas":
        return {"prompt": SFAS_SEGMENT_PROMPT, "json_kind": "list"}
    if session_type == "bmi":
        return {"prompt": BMI_SEGMENT_PROMPT, "json_kind": "list"}
    if session_type == "bai":
        return {"prompt": BAI_SEGMENT_PROMPT, "json_kind": "list"}
    raise ValueError(f"Unsupported normalized session type: {session_type}")


def build_transcript(rows: List[Tuple[int, str, str]]) -> str:
    return "\n".join(f"{idx}: [{role}] {text}" for idx, role, text in rows)


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
    candidate = search_text[start:end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        pass
    # The model sometimes drops the outer closing bracket after the last entry.
    try:
        return json.loads(candidate + ("]" if json_kind == "list" else "}"))
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


def raw_output_path_for_clean_csv(clean_csv_path: str) -> str:
    return clean_csv_path.replace(".cleaned_step4.csv", ".step4_raw.json")


def write_raw_output(path: str, records: List[Dict[str, object]]) -> None:
    """Save the model's full responses (reasoning + JSON) for later inspection."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


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
        # Segments before the first boundary (e.g. a recording that starts
        # mid-session) or a session whose model output could not be parsed.
        return {
            "session_component": UNKNOWN_COMPONENT,
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


def read_coded_segments_csv(path: str) -> List[Tuple[int, str, str]]:
    """
    Return [(segment_id, role, text), ...] sorted by segment_id.
    The component model uses these Step 3 segment ids as start_index values.
    """
    rows: List[Tuple[int, str, str]] = []
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
            role = (r.get("role") or "").strip() or "unknown"
            rows.append((segment_id, role, text))
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
    top_p: float,
    reasoning_effort: str,
    timeout: float,
) -> Tuple[str, str]:
    """Return (answer content, thinking content)."""
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
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        # Qwen3.8: thinking on for this step; sampling follows Qwen's
        # thinking-mode recommendation (temperature 1.0, top_p 0.95, top_k 20).
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": reasoning_effort},
        },
    )
    message = response.choices[0].message
    thinking = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
    return message.content or "", thinking


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

        raw_out_path = raw_output_path_for_clean_csv(clean_out_path)
        raw_records: List[Dict[str, object]] = []
        for model in args.models:
            t0 = time.time()
            try:
                for attempt in range(3):  # retry when the output is cut off or not valid JSON
                    content, thinking = await call_chat(
                        client=client,
                        model=model,
                        user_content=user_content,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        reasoning_effort=args.reasoning_effort,
                        timeout=args.timeout,
                    )
                    parsed = extract_json(content, cfg["json_kind"])
                    if parsed is not None:
                        break
                    print(f"[RETRY] {model} attempt {attempt + 1}: output not parsed")
                parsed_ok = parsed is not None
                raw_records.append({
                    "input_file": csv_path,
                    "session_type": session_type,
                    "model": model,
                    "latency_sec": round(time.time() - t0, 3),
                    "parsed_ok": parsed_ok,
                    "parsed_json": parsed,
                    "raw_output": content,
                    "thinking": thinking,
                })
                if args.write_full_output:
                    write_component_csv(full_out_path, source_rows, session_type, model, parsed_ok, parsed)
                write_clean_component_csv_from_rows(clean_out_path, source_rows, parsed, DEFAULT_CLEAN_FIELDS)
                print(f"[DONE] {model} ({round(time.time() - t0, 3)}s) parsed_ok={parsed_ok}")
            except Exception as exc:
                raw_records.append({
                    "input_file": csv_path,
                    "session_type": session_type,
                    "model": model,
                    "latency_sec": round(time.time() - t0, 3),
                    "parsed_ok": False,
                    "parsed_json": None,
                    "raw_output": None,
                    "error": repr(exc),
                })
                if args.write_full_output:
                    write_component_csv(full_out_path, source_rows, session_type, model, False, None)
                write_clean_component_csv_from_rows(clean_out_path, source_rows, None, DEFAULT_CLEAN_FIELDS)
                print(f"[ERR ] {model}: {exc}")
        write_raw_output(raw_out_path, raw_records)
        print(f"[RAW ] {raw_out_path}")


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
        help="auto infers per file from the session name",
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
    parser.add_argument("--max_tokens", type=int, default=32000, help="Includes the model's thinking tokens")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--reasoning_effort", choices=["low", "medium", "xhigh"], default="medium")
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
