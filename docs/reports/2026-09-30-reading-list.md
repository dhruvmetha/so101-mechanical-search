# Reading list for the mechanical search expert

Written 2026-09-30 by Claude for dm1487. It extends `demonstrator/RELATED_WORK.md`, and papers that file already discusses are marked "also in RELATED_WORK.md".
Every entry was checked that day against its arXiv or Crossref record plus a web search or page fetch, for title, authors, venue, year and a link that resolves.
The notes use only what I read in each abstract or paper text, and the project facts come from `docs/knowledge/state.md` and RELATED_WORK.md.

## Start here

1. **Mechanical Search: Multi-Step Retrieval of a Target Object Occluded by Clutter.** Danielczuk et al., ICRA 2019.
   <https://arxiv.org/abs/1903.01588>
   Names the problem and tests five hand-written policies that choose a push, suction or grasp each step from RGB-D,
   over 15,000 simulated and 300 real trials on heaps of 10 to 20 objects. The best passed 95%, and the action count
   grew about linearly with heap size. It is the baseline our bin extends, and our rules are stricter: no cube may leave
   the bin, and the pile is up to three layers deep.

2. **Mechanical Search on Shelves using Lateral Access X-RAY.** Huang et al., IROS 2021.
   <https://arxiv.org/abs/2011.11696>
   LAX-RAY predicts a distribution over where the hidden target sits on a shelf and pushes occluders sideways to shrink
   it, greedily or with a two-step lookahead. It got 87.3% over 800 simulated shelves and at least 80% on a Fetch. This
   is the published setup closest to our shelf task, and its distribution is a ready-made first version of the honest
   planner's "map of where the target could be". Our shelf adds toppling and a ban on touching the shelf.

3. **A Planning Framework for Non-Prehensile Manipulation under Clutter and Uncertainty.** Dogar and Srinivasa,
   Autonomous Robots 2012. <https://doi.org/10.1007/s10514-012-9306-z>
   Plans pushes and push-grasps as a combinatorial search over an action library derived from pushing mechanics,
   conservative by construction, with planning times of seconds. Pushing funnels pose uncertainty, so a push can work
   where a grasp needs a pose we lack. That is a principled rule for mixing pushes and grasps, and it applies to the 1
   in 10 box-fit poses RELATED_WORK.md measured off by up to 16 mm. Also in RELATED_WORK.md.

4. **Online Planning for Target Object Search in Clutter under Partial Observability.** Xiao et al., ICRA 2019.
   <https://doi.org/10.1109/ICRA.2019.8793494>, author copy <https://khoury.northeastern.edu/home/ycx424/icra2019.pdf>
   Casts target search in clutter as a POMDP and plans online with PA-POMCP, which scores each parameterised
   manipulation by its chance of success under the current belief. It needed fewer object moves than simpler methods.
   This is the formal version of the honest planner in state.md, with moves chosen on a belief over the red, not on the
   true pile.

5. **Integrated task and motion planning in belief space.** Kaelbling and Lozano-Pérez, IJRR 2013.
   <https://doi.org/10.1177/0278364913484072>, open copy <https://dspace.mit.edu/handle/1721.1/87038>
   Plans over beliefs by regressing goals backward, with logical conditions on beliefs as the subgoals, and shows that a
   few symbolic operators produce task-directed looking. Read it for the vocabulary. "The red's position is known to
   within a centimetre" becomes a subgoal that a look or an uncovering move achieves, which turns "looks as moves" into
   planner operators instead of rulebook prose.

6. **Online Replanning in Belief Space for Partially Observable Task and Motion Problems.** Garrett et al., ICRA 2020.
   <https://arxiv.org/abs/1911.04577>
   Determinizes action outcomes, tracks object poses with particle filters, plans with PDDLStream, then executes,
   observes and replans. Each replan first tries to keep the unexecuted tail of the last plan, which saves time and
   stops the robot undoing its own progress. This is the "plan on the belief, act once, look, replan" loop under
   discussion. Tail reuse matters in chaotic piles, where each fresh replan could pick a different cover order.

7. **Imitating Task and Motion Planning with Visuomotor Transformers.** Dalal et al., CoRL 2023.
   <https://arxiv.org/abs/2305.16309>
   OPTIMUS distills a TAMP expert that reads full sim state into a Transformer that sees fixed and wrist cameras, and
   reaches 70 to 80% on tasks with over 70 objects. Three findings carry over. Pruning demos that leave the visible
   workspace or run long gained 10%, the wrist camera helped, and task-space actions trained better than joint-space
   ones, which matters because our primitives are 30 Hz joint trajectories.

8. **Bridging the Imitation Gap by Adaptive Insubordination.** Weihs et al., NeurIPS 2021.
   <https://arxiv.org/abs/2007.12173>
   Shows that when the teacher acts on information the student cannot see, imitation averages it away, and calls this
   the imitation gap. ADVISOR weights imitation against RL state by state. This is the argument behind the rule that
   planning must not use hidden state. The planner p0.3 solved 13 of 13 seeds by digging where the red truly was, and a
   camera-only student cannot learn that from images of gray cubes.

## 1. Mechanical search and object search in clutter

- **Mechanical Search: Multi-Step Retrieval of a Target Object Occluded by Clutter.** Danielczuk et al., ICRA 2019.
  <https://arxiv.org/abs/1903.01588>
  Start here, no. 1.
- **X-Ray: Mechanical Search for an Occluded Object by Minimizing Support of Learned Occupancy Distributions.**
  Danielczuk et al., IROS 2020. <https://arxiv.org/abs/2004.09039>
  A network trained only on synthetic heaps predicts where the hidden target could be, and grasping to shrink that
  area succeeded 82% of the time in sim, 15% more often than the best baseline, with a 15% gain and 17% fewer actions
  on a YuMi. Our sim can render such training piles for free. Also in RELATED_WORK.md. The 1904.10348 in that file's
  source list is Labbé et al.'s MCTS rearrangement paper, in case it was meant to be X-Ray.
- **Mechanical Search on Shelves using Lateral Access X-RAY.** Huang et al., IROS 2021.
  <https://arxiv.org/abs/2011.11696>
  Start here, no. 2.
- **Mechanical Search on Shelves with Efficient Stacking and Destacking of Objects.** Huang et al., ISRR 2022,
  Springer proceedings 2023. <https://arxiv.org/abs/2207.02347>
  STAX-RAY adds destacking and restacking moves and an MCTS policy. The greedy policy took fewer steps, and MCTS was
  more robust to real perception noise, the same trade the shelf planner faces. The abstract gives 66 to 100% real
  success, where RELATED_WORK.md has 67.
- **Online Planning for Target Object Search in Clutter under Partial Observability.** Xiao et al., ICRA 2019.
  <https://doi.org/10.1109/ICRA.2019.8793494>
  Start here, no. 4.
- **Object search by manipulation.** Dogar et al., Autonomous Robots 2014, conference version ICRA 2013.
  <https://doi.org/10.1007/s10514-013-9372-x>
  Separates objects moved because the target may hide behind them from objects moved because they block the arm, and
  gives a greedy order that is optimal under stated conditions. Our covers and jaw-axis neighbours are exactly those
  two relations.
- **Occlusion-Aware Search for Object Retrieval in Clutter.** Bejjani et al., IROS 2021.
  <https://arxiv.org/abs/2011.03334>
  A closed-loop shelf planner with a target distribution learned from the observation stream and an RL-trained search
  heuristic. It was trained only in sim and ran in near real time on a real robot. It is the nearest learned rival to
  an honest shelf planner.

## 2. Pushing and grasping together

- **Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning.** Zeng et al.,
  IROS 2018. <https://arxiv.org/abs/1803.09956>
  Push and grasp Q-maps learned jointly, with reward only from grasps, so a push is worth only the grasp it makes
  possible. It is the push-grasp baseline most bin work compares against. Also in RELATED_WORK.md.
- **Linear Push Policies to Increase Grasp Access for Robot Bin Picking.** Danielczuk et al., CASE 2018.
  <https://doi.org/10.1109/COASE.2018.8560406>
  It scores each push by the best grasp quality before and after, a cheap rule for when our expert should push. Pushes
  lifted grasp success by over 15% where Dex-Net could not grasp, which confirms the figure RELATED_WORK.md marked
  unverified, and they target objects stuck against bin walls and corners.
- **A Planning Framework for Non-Prehensile Manipulation under Clutter and Uncertainty.** Dogar and Srinivasa,
  Autonomous Robots 2012. <https://doi.org/10.1007/s10514-012-9306-z>
  Start here, no. 3.
- **Manipulation Planning Among Movable Obstacles.** Stilman et al., ICRA 2007.
  <https://doi.org/10.1109/ROBOT.2007.363986>
  Searches backward in time, sampling the final robot action first and then the object moves it requires. That fits
  our bin. Fix the grasp on the red first, then derive which covers and neighbours must move.
- **Rearrangement planning using object-centric and robot-centric action spaces.** King et al., ICRA 2016.
  <https://doi.org/10.1109/ICRA.2016.7487583>
  One planner mixes object-centric actions, which move a chosen object, with robot-centric ones, which move the arm and
  let it touch many objects at once. A sweep is robot-centric and a pick is object-centric, so this is the planning
  frame for mixing them.

## 3. Task and motion planning under partial observability

- **Integrated Task and Motion Planning.** Garrett et al., Annual Review of Control, Robotics, and Autonomous Systems
  2021. <https://arxiv.org/abs/2010.01083>
  The standard survey. Its Table 1 sorts methods by how they order actions and satisfy constraints, and it points to
  the belief-space extensions. Read it before choosing a formulation for the expert.
- **PDDLStream: Integrating Symbolic Planners and Blackbox Samplers via Optimistic Adaptive Planning.** Garrett et al.,
  ICAPS 2020. <https://arxiv.org/abs/1802.08705>
  Wraps black-box samplers as streams inside PDDL. Our samplers, such as places, check and the sweep corridor rule,
  could plug in this way, and Garrett 2020 and Curtis 2022 both build on it. Code at
  <https://github.com/caelan/pddlstream>.
- **Hierarchical task and motion planning in the now.** Kaelbling and Lozano-Pérez, ICRA 2011.
  <https://doi.org/10.1109/ICRA.2011.5980391>, open copy <https://people.csail.mit.edu/tlp/pdf/2011/hpnICRA11Final.pdf>
  Commits to choices top-down and plans only the next steps in detail, to cut search. Cheap planning per move is what
  makes a planner expert fast.
- **Integrated task and motion planning in belief space.** Kaelbling and Lozano-Pérez, IJRR 2013.
  <https://doi.org/10.1177/0278364913484072>
  Start here, no. 5.
- **Online Replanning in Belief Space for Partially Observable Task and Motion Problems.** Garrett et al., ICRA 2020.
  <https://arxiv.org/abs/1911.04577>
  Start here, no. 6.
- **Long-Horizon Manipulation of Unknown Objects via Task and Motion Planning with Estimated Affordances.** Curtis et
  al., ICRA 2022. <https://arxiv.org/abs/2108.04145>
  PDDLStream fed by perception modules that estimate segmentation, shape and grasps of unknown objects from RGB-D, with
  no object models. It is the template for moving the expert from fitted cubes to YCB objects and toys, and for
  state.md's open question on grasps from each object's points.

## 4. Active and interactive perception

- **Interactive Perception: Leveraging Action in Perception and Perception in Action.** Bohg et al., IEEE Transactions
  on Robotics 2017. <https://arxiv.org/abs/1604.03670>
  The survey behind treating a look and an uncovering move as one kind of action, since acting creates signals a
  still camera would never get.
- **Manipulation-based active search for occluded objects.** Wong et al., ICRA 2013.
  <https://doi.org/10.1109/ICRA.2013.6630966>, open copy <https://hdl.handle.net/1721.1/90274>
  A generative model of container contents steers the search and says which occluders to move. Our pile generator is
  a known generative model, so the prior over where the red starts can come from it instead of a flat map.
- **Act to See and See to Act: POMDP planning for objects search in clutter.** Li et al., IROS 2016.
  <https://doi.org/10.1109/IROS.2016.7759839>
  A POMDP where moving objects is how the robot gathers information, made tractable with spatial constraints. It beat
  greedy methods most where multi-step manipulation was needed, which is the case for lookahead on buried starts.
- **Object Finding in Cluttered Scenes Using Interactive Perception.** Novkovic et al., ICRA 2020.
  <https://arxiv.org/abs/1911.07482>
  RL on a TSDF finds a hidden object by moving clutter, transferring from sim to a real arm at over 88%. RELATED_WORK.md
  notes that a hand-written sweep beat it on success in the same scenes.
- **Visuomotor Mechanical Search: Learning to Retrieve Target Objects in Clutter.** Kurenkov et al., IROS 2020.
  <https://arxiv.org/abs/2008.06073>
  RL that uncovers a target under a heap, made efficient by a teacher for exploration and a critic given privileged
  state, while the policy acts from images. That is one honest way to use sim state if a student is ever trained with
  RL.

## 5. Demonstrations from planners and experts

- **Imitating Task and Motion Planning with Visuomotor Transformers.** Dalal et al., CoRL 2023.
  <https://arxiv.org/abs/2305.16309>
  Start here, no. 7.
- **Guided Imitation of Task and Motion Planning.** McDonald and Hadfield-Menell, CoRL 2021.
  <https://arxiv.org/abs/2112.03386>
  An asynchronous, distributed TAMP solver makes supervision fast enough for imitation, and partly trained policies cut
  planning time by up to 2.6 times. Small observation changes can flip the plan, so the student is split into a policy
  that picks parameterised actions and one that outputs controls. Our demos are already parameterised primitives.
- **MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations.** Mandlekar et al., CoRL
  2023. <https://arxiv.org/abs/2310.17596>
  Made over 50,000 demos from about 200 human ones by moving object-centric segments to new poses, executing them and
  keeping successes. It assumes a known subtask order, which a search task lacks. Its loop could still multiply the
  late stage of a demo, such as lifting an uncovered red.
- **DART: Noise Injection for Robust Imitation Learning.** Laskey et al., CoRL 2017.
  <https://arxiv.org/abs/1703.09327>
  Injects noise into the demonstrator so the data shows recovery, and beat behaviour cloning by 62% on grasping in
  clutter. A replanning expert recovers from any perturbation, so this recovery data costs us almost nothing extra.
- **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.** Ross et al., AISTATS
  2011. <https://arxiv.org/abs/1011.0686>
  DAgger has the expert label states the student reaches. It needs an expert you can query from any state. A planner
  is cheap to query, and a Claude player at about 5M cached tokens a demo is not. The proposed stateless Claude player,
  one fresh call per move, can also be queried from any state, at a cost nobody has measured yet.
- **Learning by Cheating.** Chen et al., CoRL 2019. <https://arxiv.org/abs/1912.12294>
  A driving agent that sees the true layout teaches a vision-only agent, which reached 100% on the original CARLA
  benchmark. The teacher's extra facts, road layout and other cars, are largely visible to the camera. A buried red is
  not, so a p0.3-style teacher would hit the gap in Start here, no. 8.
- **Bridging the Imitation Gap by Adaptive Insubordination.** Weihs et al., NeurIPS 2021.
  <https://arxiv.org/abs/2007.12173>
  Start here, no. 8.

## 6. Sim realism for contact-rich manipulation

- **Convex and analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo.**
  Todorov, ICRA 2014. <https://doi.org/10.1109/ICRA.2014.6907751>
  The convex, soft contact model MuJoCo solves every step. The MuJoCo docs say soft constraints make some contact slip
  inherent, and the noslip pass that `models/stage.xml` turns on suppresses it, at some risk of instability with many
  contacts. Read it before randomizing friction or solver settings.
- **Mechanics and Planning of Manipulator Pushing Operations.** Mason, IJRR 1986.
  <https://doi.org/10.1177/027836498600500303>
  The founding analysis of pushing. Its voting theorem gives the direction a pushed object turns without knowing the
  pressure distribution under it. With friction hidden per episode, plan on push outcomes like that, which do not
  depend on it.
- **More than a Million Ways to Be Pushed: A High-Fidelity Experimental Dataset of Planar Pushing.** Yu et al., IROS
  2016. <https://arxiv.org/abs/1604.04038>
  Real pushes showing that friction varies across a surface, over time as it wears, with speed and with direction.
  Use it to size the friction randomization, and remember that one real object has no single friction coefficient.
- **Sim-to-Real Transfer of Robotic Control with Dynamics Randomization.** Peng et al., ICRA 2018.
  <https://arxiv.org/abs/1710.06537>
  Randomizes 95 parameters, puck mass and friction among them. On the real arm an LSTM reached 0.89 against 0.67 to
  0.70 for feedforward policies, and fixed puck friction in training cut it to 0.48. Hidden physics needs memory, and
  fixed latency or no sensor noise cost more still, 0.29 and 0.25.
- **RMA: Rapid Motor Adaptation for Legged Robots.** Kumar et al., RSS 2021. <https://arxiv.org/abs/2107.04034>
  Trains a policy given the true environment factors, mass and friction among them, then a module that estimates them
  from the last 0.5 s of states and actions. That is the honest pattern for hidden physics. The expert can estimate
  friction from how the last push moved the pile, and a student can learn to do the same.
