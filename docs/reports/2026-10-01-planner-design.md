# Planner design for the mechanical search expert

Written 2026-10-01 by Claude for dm1487: which planner formulation fits, which models it needs, how to combine it
with Claude, and the build order. Labels: [Key] is a paper in the source list, opened for this report; [ours: path] is
a measurement in this repo; [reasoning] is my inference, untested.

## Short answer

- Plan on a belief built from camera data, one move at a time, replanning after every look. A fixed skeleton of
  phases gives the long horizon: find the red, see it clearly, free a grasp, lift. Each decision takes the checked
  candidate with the fewest expected moves to the next phase, from calibrated success probabilities, and vetoes high
  spill risk. Not POMCP. Depth-2 search only when stuck.
- Models, by payoff: grasp outcome, spill, pose reveal, stroke outcome, hiding map. Twins rebuilt from perception
  last, and only if they pass a fidelity test. All are fit offline in simulation on other piles and read only camera
  features when deciding.
- The program plays every move and calls Claude on five triggers. Claude picks among already checked candidates from
  a one-page card. Between batches Claude edits the strategy code, which must pass a bench before use.
- First steps: freeze a bench on the 29 look_first piles; run a stateless Claude player on it; fit grasp and spill
  models.

## What the measurements already decide

- Seeing the red clearly matters most. Allowed picks lifted a well-seen red 22 of 39 times and a rough one 1 of 11
  [ours: `outputs/look_first_2026-09-30/rows.json`]. Pose quality belongs in the planner's state.
- Well-seen picks still miss 44%, 11 times closing on nothing [ours: same file]. The pick check is a hint.
- Spills ended 5 of 29 piles under the look policy, and 8 spilling strokes passed act's rule [ours: same file].
- One stroke can open the red in 25 of 29 piles, but 1 stroke in 4 works and the straight-line push report predicts
  badly: strokes moved 7 to 11 cubes and turned the red 50 deg [ours: journal 2026-09-30 20:04, 20:07]. Stroke choice
  needs a model fit to outcomes.
- A pick costs 1.9 s of CPU; arrakis has 32 shared cores [ours: `docs/knowledge/mechanical-search-learnings.md`].
  Hundreds of rollouts per decision fit, thousands do not.
- Claude per move solved 24 of 25 at about 5.1M cached tokens and 40 to 50 requests a demo [ours:
  `demonstrator/REPORT.md`; journal 2026-09-30 18:10]. Those demos used wrist depth and miss undos, so no honest
  Claude number exists on comparable piles.
- With true state, a depth-2 lookahead fixed all four seeds greedy p0.2 stuck on [ours: `docs/knowledge/planner.md`].

## 1. Which formulation

| Option | Evidence | Why not alone |
|---|---|---|
| Greedy on a learned occupancy map | X-Ray grasps the object overlapping the predicted target map most: 82% over 1,000 simulated heaps, 20 of 20 real [Danielczuk 2020] | Removes grasped objects, one-layer heaps, grasps only. Cannot value pushes, looks or spills. |
| POMCP over particles | PA-POMCP solved all 100 random tabletop scenes, moving fewer objects than greedy policies, several of which looped [Xiao 2019] | 400 particles and 600 simulations per step in a kinematic simulator. In MuJoCo, 600 two-move rollouts are about 40 CPU-minutes per decision [reasoning]. Chaotic piles make deep rollouts noisy. |
| Belief-space TAMP, determinize and replan | SS-Replan: particle filters over poses, replan after each observation, reuse the plan's unexecuted tail [Garrett 2020]. TAMPURA adds risk awareness for irreversible outcomes [Curtis 2024] | Right structure; the symbolic search is more than a task with a known phase order needs. |
| Lookahead on predicted images | LAX-RAY's two-step policy was best on 800 simulated shelves, 87.3% [Huang 2021] | Its predictions assume nothing hides behind a pushed object, and errors grew with depth. STAX-RAY's MCTS took more steps than greedy in simulation [Huang 2022]. |

The recommendation takes the third row's structure and the first row's scoring.

The belief is what `Episode.see()` returns plus: a map of floor cells where a target-sized object could hide unseen,
weighted by red pixels; the red's pose with its quality (unseen, rough, trusted, from perception's `how` field); and
counters for misses and undone spills. It holds nothing the cameras did not show.

Phases are belief conditions, as in "the red's position is known to a few millimetres", a subgoal that a look or a
revealing push achieves [Kaelbling 2013]. For a candidate that reaches the next phase with probability p, self-loop
determinization gives the expected cost c + c'(1/p - 1), with c' the cost of recovering from failure [Garrett 2020].
Add the estimated cost of the remaining phases and veto candidates whose spill probability exceeds a threshold, as
TAMPURA avoids irreversible outcomes [Curtis 2024].

A worked example with our numbers [reasoning]. A pick on a rough red, p = 0.09 and c' = 1 move, costs about 11
moves. A stroke near the red makes a rough pose trusted 54% of the time [ours: journal 2026-09-30 20:07], so revealing
costs 1/0.54, about 1.9 moves, and the trusted pick 1 + (1/0.56 - 1), about 1.8: about 3.6 in all. "Look before you grasp"
falls out of the arithmetic, which also says when a look is not worth its robot time.

Replan after every look, keeping the last decision's next step while it stays valid, which stops the planner undoing
its own progress [Garrett 2020] and helps the rule that similar situations get similar moves. When no candidate
advances a phase, score the three best first moves by the best second move each allows, as p0.3 did.

Two cautions. Planning "in the now" assumes few irreversible outcomes and loose geometry [Kaelbling 2011]; our bin has
neither, so the spill veto applies to every move. Greedy policies loop when objects block each other in order [Xiao
2019]; the stuck trigger and depth-2 search guard against it.

For other containers and objects, the phases, hiding map, height-map features and scoring carry over. The models are
refit in simulation per gripper and object set. Twins need meshes of partly seen objects, harder for YCB than cubes
[ours: `demonstrator/consult/fable-2026-09-30.md`].

## 2. The models, ranked by payoff

Inputs come only from `see()` and `check()`; training seeds are disjoint from the bench; labels are simulator
outcomes. That is honest the way X-Ray's synthetic training is: the model carries experience from other piles, no
fact about this pile the cameras missed [reasoning; Danielczuk 2020]. Start with logistic regression or boosted trees
on 10 to 20 hand features, and move to a network on height-map crops only if they saturate. For pile pushing, a
linear model beat deep models trained on the same data [Suh 2020].

1. Grasp outcome, P(red lifted | pick). Features: hit depths, fingers on the target's top, aim shift, unseen depth on
   the finger path, wall gap, pose quality. Data: picks in simulation from snapshots, under 2,000 per core-hour.
   Payoff: every success ends in a pick, and misses come from where fingers land, which the height map shows
   [ours: `outputs/miss_study_2026-09-30/`].
2. Spill risk, P(a cube leaves | stroke). Features: height of carried material, its distance to the wall ahead,
   stroke length, unseen depth in the corridor. 768 calibration strokes and 532 push_goal candidates exist [ours].
3. Pose reveal, P(red trusted after the move). The label is perception's own verdict, so no ground truth is needed.
   It is what makes looking first pay.
4. Stroke outcome, P(next phase reached | stroke). A value model, not an image predictor. DIPN predicts post-push
   images and scores them with a grasp network [B. Huang 2021], more than we need to start. High payoff for move
   count; noisy labels, since piles are chaotic.
5. Hiding map. X-Ray's label is the set of target placements that would give the same image, and a network learns it
   at 6 ms against 1.5 s [Danielczuk 2020]. For cubes the height map gives it directly, sharpened by the sliver rule
   that held in every buried episode [ours: learnings, "Seeing the pile"]. Modest payoff now (the red was listed in 32
   of 48 starts); more for buried starts and YCB.
6. Twins: seen cubes at perceived poses with noise, hidden cubes placed in unseen space and kept only if still hidden,
   as PA-POMCP generates particles [Xiao 2019], then settled. Payoff unknown; planner.md notes that a twin built from
   the same perception repeats its errors.

Hidden friction and density matter less while their drawn ranges stay narrow [ours: `docs/knowledge/state.md`].

## 3. The hybrid with Claude

The program does the routine work: observe, update the belief, generate candidates (p0.3's generator, the look_first
strokes, looks), run act's checks, score, commit, gate the export. It calls Claude when:

- Stuck: no candidate passes the veto with at least 0.15 chance of advancing a phase, or two moves advanced nothing.
- Close call: the best two candidates differ in kind or bet on different hiding spots, within the models'
  calibration error. KnowNo sets such a margin by conformal prediction so the help rate is known in advance [Ren
  2023]; there an LLM asks a human, here the program asks Claude.
- Surprise: the red moved over 2 cm unasked, a trusted red vanished, a stroke predicted safe spilled, or a pick
  predicted at 0.7 or better missed.
- Budget: moves used plus estimated moves left exceed the cap.
- Novelty, later: features outside the training range, such as a new container.

Claude sees a card of about 2,000 to 3,000 tokens [reasoning; observe printed about 750, journal 2026-09-30 18:10]:
phase and trigger, a belief note under 300 words carried between calls as Statler carries a world-state estimate
[Yoneda 2024], the red's estimate, covers and blocked ends, the top three to five checked candidates with model
numbers, moves used, and one annotated camera crop. Rubric and strategy sit in a cached system prompt. Claude returns
JSON: a candidate id, "look" or "stop", a one-line reason, the updated note, optionally a proposed rule. A move Claude
wants that is not listed gets generated and checked like any other; act's refusals always apply. LLMs propose, sound
verifiers decide [Kambhampati 2024]; Text2Motion, which checks LLM plans for geometric feasibility, solved 82% where
earlier language planners solved 13% [Lin 2023].

Keep Claude out of the inner loop: LLM-MCTS called the LLM inside tree search at 68 s per decision, against 1.2 s for
the LLM as a plain policy [Zhao 2023]. Spatial guesses from Claude enter as a soft prior with a floor, so a wrong
guess delays the search instead of excluding the true spot [Singha 2026; Sharma 2023]. Gray cubes offer no semantic
prior; toys and YCB objects will.

Consistency comes from five things: a deterministic program; a cache of Claude's answers keyed by a coarse signature
(phase, pose quality, covers, blocked ends, candidate kinds), reused without a call, as plan-template caching halved
agent costs elsewhere [Zhang 2025]; rubric, strategy and models frozen per batch; agreement measured by re-asking
sampled cards in fresh calls; and promotion into code of any choice Claude makes the same way three times.

Two designs. As strategy author, Claude edits the candidate generator and scoring, validated on training piles with
failures fed back. GPT-4 wrote generalized planners as programs debugged against training tasks, often from two
[Silver 2024], and LLM-written search code solved four problem sets with O(1) calls per domain [Katz 2024]. Our own
warning: common sense written as rules without validation picked the working stroke in 8 of 29 piles [ours:
`outputs/push_goal_2026-09-30/choice.json`]. As arbiter, Claude picks among checked candidates on triggers. Use
both: author between batches, arbiter for the residue.

Calls per demo are unknown until measured (state.md lists it as open). If triggers fire on one to three of about
eight decisions, at about 5,000 new input tokens plus a cached prefix and 1,000 to 2,000 output tokens each, a demo
costs under 30,000 tokens, under 1% of 5.1M [reasoning].

## 4. Build order

1. Freeze the bench, half a day: the 29 look_first piles at 5 and 15 moves, 10 buried seeds at 15, disjoint training
   seeds, one script for solved, moves, robot time, spills, kept misses, resets, calls, tokens. Baselines: once 3,
   loop 9, look 11, union of rule policies 14 of 29 [ours: look_first rows].
2. Stateless Claude on the bench, about a day: one fresh call per move with the card, act's refusals in force. This
   is the queued "Claude plays piles itself" step in card form, and gives the honest Claude number. At 12 or fewer of
   29, Claude's judgment is not the bottleneck and models matter more; at 20 or more, the arbiter has room.
3. Grasp and spill models, one to two days. Pass: predicted rates within 10 points of observed per bin; refusing
   low-p picks cuts kept misses without losing solved piles. Stop if features separate outcomes poorly (AUC under
   0.7): the gain then lies in better looks.
4. Reveal and stroke models, two days. Pass: the chosen first stroke plus a real pick clearly beats once's 3 of 29.
5. Planner p1, two to three days, no Claude; truth accessors raise inside the planner process. Pass: 15 or more of
   29 at 5 moves, and buried seeds near b5000's and b5002's 11 and 13 moves [ours: journal]. Under 11 of 29, move
   step 7 forward.
6. The arbiter, one day. Pass: solved within one pile of the better of steps 2 and 5, at a fifth of step 2's tokens
   or less, with 80% agreement on repeated cards; below that, keep only strategy edits.
7. Twin fidelity, only if p1 stalls on over a fifth of buried seeds: on 100 recorded moves, compare 8 twins'
   predicted outcomes with what happened, and adopt twins for depth-2 search only if they beat step 4's models.
8. The strategy loop, then demos: Claude proposes code edits from the trigger log, shipped only if the bench shows no
   regressions.

Stop building the hybrid if p1 alone matches stateless Claude on most piles. Return to a Claude player if the hybrid
needs over five calls a demo, since a stateless player was estimated at 150,000 tokens a demo [ours: journal
2026-09-30 19:10, an estimate].

## Where the literature has no answer

- I found no mechanical search in a layered pile where every object must stay in the container. X-Ray removes
  non-target objects [Danielczuk 2020]; RELATED_WORK.md found the same gap.
- I found no measurement of when a geometric planner should call an LLM; KnowNo covers the inverse.
- No paper reports how well perception-built twins predict pushes in a pile.
- Nothing addresses keeping an LLM teacher consistent across calls for imitation data. The five measures above are
  untested.

## Sources

- [Curtis 2024] Curtis et al., TAMPURA, arXiv 2024. <https://arxiv.org/abs/2403.10454>
- [Danielczuk 2020] Danielczuk et al., X-Ray, IROS 2020. <https://arxiv.org/abs/2004.09039>
- [Garrett 2020] Garrett et al., Online Replanning in Belief Space, ICRA 2020. <https://arxiv.org/abs/1911.04577>
- [B. Huang 2021] B. Huang et al., DIPN, ICRA 2021. <https://arxiv.org/abs/2011.04692>
- [Huang 2021] H. Huang et al., LAX-RAY, IROS 2021. <https://arxiv.org/abs/2011.11696>
- [Huang 2022] H. Huang et al., STAX-RAY, ISRR 2022. <https://arxiv.org/abs/2207.02347>
- [Kaelbling 2011] Kaelbling and Lozano-Pérez, Hierarchical TAMP in the now, ICRA 2011.
  <https://people.csail.mit.edu/tlp/pdf/2011/hpnICRA11Final.pdf>
- [Kaelbling 2013] Kaelbling and Lozano-Pérez, TAMP in belief space, IJRR 2013, abstract.
  <https://dspace.mit.edu/handle/1721.1/87038>
- [Kambhampati 2024] Kambhampati et al., LLM-Modulo, ICML 2024. <https://arxiv.org/abs/2402.01817>
- [Katz 2024] Katz et al., Thought of Search, NeurIPS 2024. <https://arxiv.org/abs/2404.11833>
- [Lin 2023] Lin et al., Text2Motion, Autonomous Robots 2023. <https://arxiv.org/abs/2303.12153>
- [Ren 2023] Ren et al., KnowNo, CoRL 2023. <https://arxiv.org/abs/2307.01928>
- [Sharma 2023] Sharma et al., Semantic Mechanical Search, CoRL 2023. <https://arxiv.org/abs/2302.12915>
- [Silver 2024] Silver et al., Generalized Planning in PDDL Domains with LLMs, AAAI 2024.
  <https://arxiv.org/abs/2305.11014>
- [Singha 2026] Singha et al., Imagine-TAMP, arXiv, September 2026. <https://arxiv.org/abs/2609.20396>
- [Suh 2020] Suh and Tedrake, Linear models for visual foresight in pile manipulation, WAFR 2020.
  <https://arxiv.org/abs/2002.09093>
- [Xiao 2019] Xiao et al., Target Object Search in Clutter under Partial Observability, ICRA 2019.
  <https://khoury.northeastern.edu/home/ycx424/icra2019.pdf>
- [Yoneda 2024] Yoneda et al., Statler, ICRA 2024. <https://arxiv.org/abs/2306.17840>
- [Zhang 2025] Zhang et al., Agentic Plan Caching, NeurIPS 2025. <https://arxiv.org/abs/2506.14852>
- [Zhao 2023] Zhao, Lee and Hsu, LLMs as Commonsense Knowledge for Task Planning (LLM-MCTS), NeurIPS 2023.
  <https://arxiv.org/abs/2305.14078>

Missing from the reading list and worth adding: Imagine-TAMP, the closest published design to this one (a VLM-shaped
particle belief over the hidden target, determinize and replan, plan costs that include visibility), then TAMPURA,
Semantic Mechanical Search, KnowNo, LLM-Modulo, Silver 2024, Thought of Search, LLM-MCTS, Statler, plan caching, and
Suh and Tedrake. Also read, not cited above: ThinkGrasp (GPT-4o picks which occluder to remove each step, CoRL 2024,
<https://arxiv.org/abs/2407.11298>) and LLM-written POMDP models checked against data (Curtis et al., arXiv 2025,
<https://arxiv.org/abs/2505.02216>).
