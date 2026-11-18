# Contributing

Questions, reproducible bug reports, numerical corrections, and focused improvements are welcome in English or Chinese. Read the [README](README.md) or [中文说明](README.zh-CN.md) and run the existing application before changing its behavior.

## Report a problem

Open an [issue](https://github.com/AstonDky/Math_in_RL_Visual/issues) with the commit, operating system, Python version, selected algorithm, relevant `RLAlgorithmConfig` fields, and map/reward settings. Include the smallest sequence that reproduces the problem, the expected behavior, and the actual result. For numerical errors, give a small transition or trajectory with the expected update. Include tracebacks as text.

Do not upload personal paths, credentials, or unrelated files. A checkpoint is not needed for most reports; prefer a short example that constructs its state. Checkpoints use pickle and should only be loaded from trusted sources.

## Develop locally

Fork the repository, create a branch, and install `requirements.txt` in a separate Python 3.12 environment. Work from the repository root.

```bash
python -m unittest discover -s tests -v
git diff --check
```

Tests use Python's standard `unittest` and offscreen Qt, so no additional testing package or visible desktop is needed. The test suite includes short runs of all exported algorithms and targeted regressions for numerical state, DQN target updates, checkpoints, and UI session controls. It does not replace convergence studies.

For a UI change, also launch `python main.py` in a desktop session and check the affected control. Use disposable training data: startup mode `"restart"` and the Restart button replace the selected algorithm's stored run. Keep checkpoints, TensorBoard output, virtual environments, and generated caches out of commits.

## Change an algorithm

Keep numerical functions in `algorithms/`, without Qt, logging, or checkpoint code. The adapter supplies the environment, state, and helpers. Its signature rules are in `core/algorithm_adapters.py`; function exports are in `algorithms/__init__.py`. New state that must survive between calls also needs adapter and checkpoint coverage.

Preserve the relationship between mathematical quantities and their implementation. Explain any change to the learning rule, behavior policy, terminal handling, or approximation in the pull request, with a source or a worked example. Add a small regression for a corrected numerical result, not just a check that the program runs. Use seeded, short examples; do not make routine CI train to convergence.

## Submit a pull request

Describe the problem, resulting behavior, and checks you ran. Keep changes focused and update both README languages when user-facing behavior changes. Note tests or platforms you could not check. Preserve existing author credits and license notices; identify the source and license of any imported code or assets. Project contributions are submitted under the [MIT License](LICENSE), while dependencies retain their own licenses.
