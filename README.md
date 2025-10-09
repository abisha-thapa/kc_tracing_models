# Understanding and Modeling Math Strategy Use in Intelligent Tutoring Systems

We investigate how students learn to apply context-specific math strategies by analyzing data from MATHia (a widely used Intelligent Tutoring System) collected from 655 schools in USA. In particular, we focus on a set of lessons designed to teach ratios and proportions where students learn multiple strategies individually, and then are presented with lessons in which they are presented with options to make a choice between strategies. Initially, students complete workspaces (a term used for a lesson in Mathia) where they learn to apply 2 strategies independently, i.e., 1. Equivalent Ratios (pre-requisite) and 2. Means & Extremes (pre-requisite). After this, they attempt a workspace called _Calculating Percent Change and Final Amounts_ where they are presented with a choice, i.e., there are two optional tasks that are presented to them one of which scaffolds the problem using the Equivalent Ratios (ER) and the other scaffolds the Means & Extremes (ME) strategy. 

## Knowledge Tracing

Learning to _recognize_ which strategy to apply is a skill that students need to master over time. We encode this skill as a hidden state in Bayesian Knowledge Tracing (BKT) and Deep Knowledge Tracing (DKT) models to predict future use of optimal strategies. Specifically, we model two skills that represent the students’ knowledge to recognize the optimality of the corresponding strategies in the right context, namely __Recognize-ER__ and __Recognize-ME__.

> When ER is applied to a problem that is optimal for ER, we assign that observation as a positive label for Recognize-ER else we assign a negative label for that observation.

> Similarly, we assign labels for Recognize-ME based on whether we observe ME was applied to a problem where it was the optimal strategy.


For BKT to predict the use of future optimal strategy on sample test data, use the following code
> python trace_models/bkt_model.py 

## Generation using LLMs

We evaluated the effect _adaptive supports_ from the data of this workspace. Using LLMs to generate student strategies can significantly scale-up testing and validation of hypotheses by reducing/eliminating the need for large-scale data collection. To evaluate its feasibility, we task the LLM to generate student strategies and analyze the effect of the _adaptive supports_, and compare this with our observations from the real data.

- __full_prompt.md__ : The full prompt is available here.
- __LLM_responses__: The repsonses from LLM is inside this folder. 
