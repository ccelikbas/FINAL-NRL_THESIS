"""training/ — modular replacement for run_multiple_curriculums.py.

Five helper modules define WHAT to train; the training ENGINE and the run
control live in ../train_master.py (which imports these):

    scenarios.py        S1/S2 worlds + the four curriculums + the shared `Job` type
    job_complete_s1.py  }  one declarative Job spec per (model × scenario)
    job_baseline_s1.py  }  combination — imported by train_master.py, not run
    job_complete_s2.py  }  on their own.
    job_baseline_s2.py  }

Edit train_master.py to choose which jobs to train, how many times, and from
which checkpoints. The original run_multiple_curriculums.py is kept unchanged.
"""
