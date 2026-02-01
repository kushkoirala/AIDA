# AIDA: Updated Literature Review and Background

**Kushal Koirala | Wichita State University | PhD Thesis Proposal**

---

## Introduction

The literature review for AIDA spans six technical pillars that collectively define the research landscape: (1) Inverse Reinforcement Learning for intent inference, (2) Control Barrier Functions for safety-critical autonomy, (3) Large Language Models for autonomous flight systems, (4) Pilot intent inference and human--AI teaming, (5) Flight envelope protection and certification, and (6) Speech recognition and voice technology. For each area, we identify the foundational work, the current state of the art, and the specific research gaps that AIDA addresses.

---

## 1. Inverse Reinforcement Learning and Bayesian Reward Inference

Inverse Reinforcement Learning (IRL) addresses the problem of recovering a reward function from observed behavior, rather than learning a policy directly. The foundational formulation by **Ng and Russell (2000)** [1] characterized the set of reward functions for which a given policy is optimal and presented three algorithms for IRL under different observability conditions. A key challenge identified was degeneracy---multiple reward functions can explain the same observed behavior.

**Ramachandran and Amir (2007)** [2] introduced Bayesian Inverse Reinforcement Learning (BIRL), which resolves the degeneracy problem by computing a posterior distribution over reward functions using Markov Chain Monte Carlo (MCMC) sampling. BIRL models the likelihood of observed actions via a Boltzmann distribution parameterized by an inverse temperature $\beta$ and state-action Q-values. Unlike prior IRL methods, BIRL does not require a fully specified optimal policy, can incorporate partial demonstrations, and handles sub-optimal behavior---properties essential for modeling human pilots who deviate from optimality under workload and stress.

**Ziebart et al. (2008)** [3] proposed Maximum Entropy IRL, which resolves ambiguity by selecting the reward function that induces the maximum-entropy distribution over trajectories consistent with observed behavior. This approach provides globally normalized distributions over decision sequences and was validated on real-world driving and navigation tasks, demonstrating applicability to noisy human demonstrations.

More recently, **Bayesian IRL-based reward learning for automated driving** has been explored directly. Huang et al. (2024) [4] applied BIRL to learn driving reward functions from human demonstrations, showing that the Bayesian posterior over reward weights can capture multi-objective driving preferences (safety, comfort, efficiency) in a principled way. A comprehensive survey by **Arora and Doshi (2021)** [5] cataloged the evolution of IRL techniques from linear reward assumptions through deep IRL methods, identifying the gap between theoretical frameworks and real-time deployment in safety-critical cyber-physical systems.

**Gap addressed by AIDA:** Existing BIRL work targets offline reward inference from collected datasets. AIDA implements online, streaming BIRL with a lightweight MCMC sampler (Policy Walk) running at 1 Hz within a 50 Hz flight control loop, using the posterior entropy as a real-time uncertainty signal that modulates downstream safety constraints. No prior work couples BIRL posterior entropy directly to a control barrier function for adaptive safety enforcement.

---

## 2. Control Barrier Functions for Safety-Critical Autonomous Systems

Control Barrier Functions (CBFs) have emerged as the dominant formal framework for enforcing safety in autonomous systems. **Ames, Xu, Grizzle, and Tabuada (2017)** [6] introduced the CBF-QP formulation that unifies safety constraints (expressed as barrier functions) with performance objectives (expressed as control Lyapunov functions) through real-time quadratic programming. The key insight is that forward invariance of a safe set can be guaranteed by enforcing a Lyapunov-like condition $\dot{h}(x) + \alpha(h(x)) \geq 0$ as a linear constraint on the control input, yielding a minimally invasive safety filter.

**Ames et al. (2019)** [7] provided a comprehensive survey of CBF theory and applications, extending the framework to handle higher relative-degree safety constraints and demonstrating applications in bipedal locomotion and adaptive cruise control. This survey established CBFs as the standard tool for safety-critical control synthesis.

Recent advances have addressed key practical limitations. **Adaptive CBFs (Taylor and Ames, 2019)** [8] handle parametric model uncertainty by adapting barrier function parameters online, maintaining safety guarantees even when the system model is imprecise. **Cohen and Belta (2024)** [9] presented advances addressing time-varying constraints, input saturation, and disturbance rejection, while **Singletary et al. (2024)** [10] developed reduced-order CBF methods that allow safety certification for complex high-dimensional systems through tractable lower-dimensional models.

The intersection of CBFs with learning has been surveyed by **Dawson et al. (2024)** [11], who cataloged approaches for learning barrier functions from data, including neural CBFs and CBFs integrated with reinforcement learning. **Cheng et al. (2019)** [12] demonstrated safe RL using CBFs as real-time safety filters layered on top of learned policies, showing that the QP-based safety filter can correct unsafe actions from a neural network policy without significantly degrading performance.

**Gap addressed by AIDA:** Prior CBF work uses fixed or parametrically adaptive barrier functions. AIDA introduces *entropy-modulated* CBFs where the safe set dynamically contracts or expands based on the BIRL posterior entropy---a measure of how well the system understands the pilot's intent. When intent is ambiguous (high entropy), barriers tighten conservatively; when intent is clear (low entropy), barriers relax to allow the pilot greater authority. This coupling of epistemic uncertainty from reward inference to the safety layer is novel.

---

## 3. Large Language Models for Autonomous Flight and Robotics

LLMs have rapidly expanded from text generation into embodied autonomous systems. **A comprehensive survey by Hu et al. (2023, updated 2025)** [13] covers the integration of LLMs into robotic autonomy across planning, perception, and control. The survey identifies that while LLMs excel at high-level reasoning and natural language understanding, they lack physical intuition---they do not inherently understand geometry, dynamics, or sensor semantics, which can lead to impractical plans when applied directly to control.

In the aviation domain specifically, several systems have explored LLM integration:

- **VOICI Project (EU H2020)** [14] aimed to integrate voice interactions toward a "Natural Crew Assistant," primarily reducing crew workload through structured voice commands rather than full AI-driven decision-making.
- **AviationGPT (Wang et al., 2024)** [15] fine-tuned an open-source LLM on aviation-specific corpora for operational (non-critical) tasks such as airport operations and traffic flow management.
- **Flight Arrival Scheduling via LLM (Zhou et al., 2024)** [16] demonstrated fine-tuned LLMs on historical scheduling data to improve arrival sequences, showing that LLMs can handle structured aviation decision problems.
- **Next-Generation LLM for UAV (Liang et al., 2025)** [17] identified a significant research gap: most LLM-for-UAV studies address isolated aspects (planning *or* control *or* language parsing), but no comprehensive system progresses from natural language input through path planning to actual vehicle control in a closed loop.

The broader LLM-for-robotics literature reveals a common architectural pattern: LLMs serve as high-level planners that invoke pre-defined low-level skills, with the LLM output constrained to a finite action vocabulary. **Agentic AI systems (2025)** [18] are beginning to close the loop, but safety assurance for LLM-generated actions in physical systems remains an open problem.

**Gap addressed by AIDA:** AIDA places the LLM at the pilot interface layer---interpreting natural language flight commands---but does *not* allow the LLM to directly command flight controls. Instead, LLM outputs are validated through a multi-stage pipeline: schema validation, constraint checking against the current flight envelope, Bayesian anomaly detection using learned priors from flight phase distributions, and finally the BIRL+CBF safety layer. This architecture addresses the fundamental reliability concern with LLMs in safety-critical systems by treating LLM output as an *untrusted input* that must pass through formal safety verification before reaching the flight controller.

---

## 4. Pilot Intent Inference and Human--AI Teaming

The problem of inferring pilot intent from observations has been studied through multiple methodological lenses. **Dong (2023)** [19] modeled single-pilot intention recognition using BiLSTM networks with attention mechanisms, encoding pilot interaction data as time-series features for intent classification. This work highlighted that misalignment between pilot intent and automation behavior is a primary safety hazard in single-pilot operations.

**Bayesian approaches** to intent inference have proven effective for modeling uncertainty. **Javdani et al. (2018)** [20] formulated intent inference in shared autonomy as recursive Bayesian filtering in a goal-conditioned Markov model, fusing multiple non-verbal observations to probabilistically reason about intended goals. The human agent's behavior is modeled as goal-directed with adjustable rationality---a formulation closely related to the Boltzmann policy model used in BIRL.

**Li et al. (2024)** [21] introduced an integrated framework for intent modeling and inference for autonomous and piloted aerial systems, using the Interacting Multiple Model (IMM) filter for feature extraction and attention-based bi-directional LSTM for intent classification. This work defined intent mathematically through critical waypoint patterns and associated motion processes.

On the human-AI teaming front, **MIT's Air-Guardian system (Schiemer et al., 2023)** [22] demonstrated a shared-control paradigm where the system uses eye-tracking to infer pilot attention and intervenes based on attention divergence rather than only during safety breaches. This proactive intervention model parallels AIDA's approach of modulating safety constraints based on intent clarity rather than waiting for envelope violations.

The broader field of **Human-AI Teaming (HAT)** in aviation has been surveyed by **Moreira et al. (2025)** [23], who define requirements for "Intelligent Assistants" that differentiate from conventional automation by having agency---the ability to form goals, make decisions, and execute them collaboratively. Key requirements include calibrated confidence sharing, transparent decision rationale, and the ability to gracefully degrade to simpler modes when the AI's understanding is uncertain.

**Gap addressed by AIDA:** Existing pilot intent inference systems operate as separate modules providing classification outputs. AIDA unifies intent inference (BIRL reward weights) with safety enforcement (CBF barrier modulation) through a single information-theoretic quantity: posterior entropy. This creates a closed-loop system where the quality of intent understanding directly controls the degree of autonomous authority---high confidence enables permissive operation, while ambiguity triggers conservative safety boundaries and advisory holds requiring pilot confirmation.

---

## 5. Flight Envelope Protection and Certification Challenges

Modern fly-by-wire aircraft implement flight envelope protection as a core safety feature. **Airbus introduced full flight-envelope protection on the A320 in 1988** [24], embedding protections within flight control laws including high angle-of-attack (stall) protection, overspeed protection, pitch and bank attitude limits, and load factor protection. These protections operate as hard constraints that the pilot cannot override in normal law---a design philosophy that directly informs CBF-based safety layers.

The certification landscape for AI in aviation presents significant challenges. **DO-178C** [25], the standard for airborne software certification, requires deterministic behavior---a property that current AI/ML systems cannot guarantee. **EASA's AI trustworthiness framework** [26] has adopted an incremental approach to different autonomy levels, recognizing that current certification standards are not fully applicable to AI technologies. **Kaakai et al. (2024)** [27] surveyed the challenges of certifying airborne AI, noting that active AI/ML is currently not permitted on commercial aircraft if logic decisions change in real time; only ground-based preflight tuning is allowed.

This creates a tension: the proven safety benefits of envelope protection (as demonstrated by Airbus's safety record) suggest that AI-based safety layers could improve aviation safety, but the certification frameworks lag the technology.

**Gap addressed by AIDA:** AIDA's CBF safety layer is designed as a *certifiable* safety filter. Unlike neural network-based approaches, the CBF-QP formulation produces provably safe controls via convex optimization with analytical barrier functions. The entropy modulation adds conservatism under uncertainty rather than relaxing constraints---a fail-safe property aligned with certification requirements. The architecture separates the non-certifiable component (LLM) from the certifiable safety layer (CBF-QP), providing a path toward incremental certification.

---

## 6. Speech Recognition and Voice Technology

Efforts by the FAA and industry dating to the 1990s tested early speech-to-text systems for cockpit tasks. **Mogford et al. (1997)** [28] evaluated voice technology for ATC applications, identifying accuracy under noise as the primary barrier. Modern neural network-based ASR has significantly improved accuracy, with **Badrinath and Balakrishnan (2022)** [29] demonstrating character-level transcription of ATC voice communications using deep neural networks, and **Kopad (2021)** [30] showing feasibility of real-time ATC instruction parsing.

The **VOICI Project** [14] advanced the concept of a "Natural Crew Assistant" integrating voice interactions, though focused on structured commands rather than open-ended AI reasoning. Recent work on **integrating LLMs into robotic voice pipelines (MDPI, 2025)** [31] reviews how LLMs can translate high-level voice commands into low-level control signals, supporting semantic planning and adaptive execution.

**Gap addressed by AIDA:** AIDA uses an edge-deployed LLM (xLAM-2-8B) as the natural language understanding layer, but adds a critical validation pipeline between language understanding and flight control execution. Voice/text commands are parsed into structured `FlightCommand` objects, validated against the current flight state using Bayesian anomaly detection with learned phase-specific priors, and filtered through the CBF safety layer before reaching the controller. This multi-stage validation addresses the core concern of misrecognition or misinterpretation in a safety-critical context.

---

## Summary of Research Gaps and AIDA's Position

| Research Area | State of the Art | Gap | AIDA Contribution |
|---|---|---|---|
| BIRL / IRL | Offline reward inference from datasets | No real-time, streaming BIRL coupled to control | Online BIRL at 1 Hz with entropy as live uncertainty signal |
| CBFs | Fixed or parametrically adaptive barriers | No coupling of epistemic uncertainty to safety set | Entropy-modulated CBFs: safe set scales with intent clarity |
| LLMs in Aviation | Advisory, non-flight-critical applications | No validated LLM-to-controller pipeline with formal safety | Multi-stage validation: schema + Bayesian + CBF filtering |
| Pilot Intent | Separate classification modules | No unified intent-to-safety closed loop | BIRL entropy directly modulates CBF barrier functions |
| Envelope Protection | Hard-coded Airbus-style normal law | No adaptive protection based on AI confidence | CBF tightens under uncertainty, relaxes under clear intent |
| Voice/NLU | ASR for ATC transcription | No validated NL command execution in flight-critical loop | Edge LLM + Bayesian validation + advisory hold system |

---

## References

[1] A. Y. Ng and S. J. Russell, "Algorithms for Inverse Reinforcement Learning," in *Proc. 17th International Conference on Machine Learning (ICML)*, pp. 663--670, 2000.

[2] D. Ramachandran and E. Amir, "Bayesian Inverse Reinforcement Learning," in *Proc. 20th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 2586--2591, 2007.

[3] B. D. Ziebart, A. L. Maas, J. A. Bagnell, and A. K. Dey, "Maximum Entropy Inverse Reinforcement Learning," in *Proc. AAAI Conference on Artificial Intelligence*, pp. 1433--1438, 2008.

[4] Bayesian Inverse Reinforcement Learning-based Reward Learning for Automated Driving, *Journal of Mechanical Engineering*, vol. 60, no. 10, pp. 245--260, 2024.

[5] S. Arora and P. Doshi, "A Survey of Inverse Reinforcement Learning: Challenges, Methods and Progress," *Artificial Intelligence Review*, vol. 54, pp. 5195--5272, Springer, 2021.

[6] A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada, "Control Barrier Function Based Quadratic Programs for Safety Critical Systems," *IEEE Transactions on Automatic Control*, vol. 62, no. 8, pp. 3861--3876, 2017.

[7] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, "Control Barrier Functions: Theory and Applications," in *Proc. 18th European Control Conference (ECC)*, pp. 3420--3431, 2019.

[8] A. J. Taylor and A. D. Ames, "Adaptive Safety with Control Barrier Functions," in *Proc. American Control Conference (ACC)*, 2019. arXiv:1910.00555.

[9] M. H. Cohen and C. Belta, "Advances in the Theory of Control Barrier Functions," *Annual Reviews in Control*, vol. 57, 2024.

[10] A. Singletary et al., "Safety-Critical Control for Autonomous Systems: Control Barrier Functions via Reduced-Order Models," *Annual Reviews in Control*, vol. 58, 2024.

[11] C. Dawson et al., "Learning Control Barrier Functions and Their Application in Reinforcement Learning: A Survey," arXiv:2404.16879, 2024.

[12] R. Cheng, G. Orosz, R. M. Murray, and J. W. Burdick, "End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks," in *Proc. AAAI*, 2019.

[13] Y. Hu et al., "Large Language Models for Robotics: A Survey," arXiv:2311.07226, updated Dec. 2025.

[14] "Solutions for Voice Interaction Towards Natural Crew Assistant," VOICI Project, EU H2020, CORDIS. Available: https://cordis.europa.eu/project/id/785401

[15] L. Wang, J. Chou, X. Zhou, A. Tien, and D. M. Baumgartner, "AviationGPT: A Large Language Model for the Aviation Domain," 2024.

[16] W. Zhou, J. Wang, L. Zhu, Y. Wang, and Y. Ji, "Flight Arrival Scheduling via Large Language Model," *Aerospace*, vol. 11, no. 10, Oct. 2024.

[17] Q. Liang et al., "Next-Generation LLM for UAV: From Natural Language to Autonomous Flight," Purdue University, arXiv, Oct. 2025.

[18] "Agentic LLM-based Robotic Systems for Real-World Applications: A Review," *Frontiers in Robotics and AI*, PMC, 2025.

[19] X. Dong, "Analysis of Single-Pilot Intention Modeling in Commercial Aviation," *International Journal of Aerospace Engineering*, Wiley, 2023.

[20] S. Javdani, S. S. Srinivasa, and J. A. Bagnell, "Shared Autonomy via Hindsight Optimization for Teleoperation and Beyond," *Int. Journal of Robotics Research*, 2018.

[21] Y. Li et al., "An Intent Modeling and Inference Framework for Autonomous and Remotely Piloted Aerial Systems," arXiv:2409.08472, 2024.

[22] M. Schiemer et al., "Air-Guardian: AI Copilot Enhances Human Precision for Safer Aviation," MIT CSAIL, 2023.

[23] C. Moreira et al., "Human Factors Requirements for Human-AI Teaming in Aviation," *MDPI Future Transportation*, vol. 5, no. 2, 2025.

[24] "Safety Innovation #7: Flight Envelope Protection," Airbus Newsroom, Feb. 2023. Available: https://www.airbus.com/en/newsroom/stories/2023-02-safety-innovation-7-flight-envelope-protection

[25] RTCA DO-178C, "Software Considerations in Airborne Systems and Equipment Certification," 2012.

[26] EASA, "Artificial Intelligence Roadmap: A Human-Centric Approach to AI in Aviation," v2.0, 2023.

[27] F. Kaakai et al., "ML Meets Aerospace: Challenges of Certifying Airborne AI," *Frontiers in Aerospace Engineering*, vol. 3, 2024.

[28] R. Mogford, A. Rosiles, D. Wagner, and K. Allendoerfer, "Voice Technology Study Report," FAA, Dec. 1997.

[29] S. Badrinath and H. Balakrishnan, "Automatic Speech Recognition for Air Traffic Control Communications," *Transportation Research Record*, vol. 2676, no. 1, pp. 798--810, Jan. 2022.

[30] H. Kopad, "Automatic Speech Recognition and Understanding of ATC Voice Communications," presented at ATIEC 2021, Sep. 2021.

[31] "Integrating Large Language Models into Robotic Autonomy: A Review of Motion, Voice, and Training Pipelines," *MDPI AI*, vol. 6, no. 7, July 2025.

[32] S. Huang, R. S. H. Teo, and K. K. Tan, "Collision Avoidance of Multi Unmanned Aerial Vehicles: A Review," *Annual Reviews in Control*, vol. 48, pp. 147--164, 2019.

[33] P. Abbeel and A. Y. Ng, "Apprenticeship Learning via Inverse Reinforcement Learning," in *Proc. 21st International Conference on Machine Learning (ICML)*, ACM, 2004.

[34] NASA TLX (Task Load Index), NASA Ames Research Center. Available: https://humansystems.arc.nasa.gov/groups/tlx/
