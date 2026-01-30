# AISE W18D4 – Distributed Training (Sports Data)

## Overview
This repository is for the Week 18 Day 4 after-class assignment focused on
**distributed training concepts**.

The goal of this project is to build a **scale-ready training loop** that
demonstrates:
- Data Distributed Parallel (DDP)
- Gradient accumulation
- Reproducibility and scaling evidence

A sports-data scenario is used to make the scaling concepts more intuitive
and easier to reason about.

Each training example represents one NBA team’s performance in a single game,
labeled as a win or loss.

## Assignment Goals
- Start from a single-process training loop
- Add gradient accumulation to handle memory constraints
- Add DDP structure to enable data parallelism
- Document scaling decisions, risks, and known distributed failure modes

## Evidence Artifacts
This repository will include:
- `metrics.csv` – training metrics and effective batch size
- `SCALE_PLAN.md` – scaling strategy and decision rationale
- `REPRO.md` – reproducibility instructions
- `KNOWN_ISSUES.md` – common distributed training failures and mitigations

## Status
Project setup and planning phase.