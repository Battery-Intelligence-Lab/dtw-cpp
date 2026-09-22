# DTWC++ 2.0 — Charter

Volkan's instructions, verbatim. Agents do not edit the quoted text; new instructions are appended
with a date. How they are turned into a design is in `design.md` §1; how they are scheduled is in
`PLAN.md`.

## 2026-09-21 — Design 2.0 branch

> This is a branch for Design 2.0, so see the existing design, see good design principles, and see
> what is done and create a design plan. Delete obsolete bits. PLAN.md and other things regarding
> the CLAUDE should be moved inside .claude folder. Create me a nice PLAN.md with all dependencies
> etc. You can use doxygen. This is a new computer so you may need to install apple clang etc. with
> brew. Let's focus on design then Opus 5.1 max can implement but since this is a big repository, I
> think it is best to get a map of what is what and then work on the map to see how we can improve
> the flow and design in terms of grand scheme. Otherwise we will always mess up the context.

> I think our previous design already works nicely, so people are using it. I don't say we cannot
> break things but we need to have a solid reason to break things. Eventually get a robust
> interface for all programming languages, with minimal or no overhead, then support all current
> and future features. I mean sometimes I use indices etc. do not to hold battery data etc. these
> kind of stuff. Or you could improve things to make parallelisation easier and/or GPU fitting
> easier. Increase accuracy, control if the algorithms are running, no useless tests just sake of
> having tests. I really like the having problem setting kind of thing like
>
> problem (device=gpu) or device=hpc,  device=cpu type of things.
>
> So by having credentials etc. the code could work on all places like SLURM type of places.
>
> Or maybe we could also consider generating SIMD code easier? Instead of just checking how many
> seconds it takes it to run, you could also compile and see ASM code for some bits and see if we
> are correctly generating SIMD-compatible ASM for some functions. Like FFMPEG mindset.

## Earlier — the 2.0 refactor brief (moved here from `TODO.md`, unchanged)

> Now we are doing a huge-refactor and upgrade our DTWC++ library.
>
> I want a very detailed analysis and a plan for Opus 4.8 -effort=xhigh to implement. Please you do
> yourself do not implement the plan but call it as the subagent to implement. Your output tokens
> are very valuable. So you need to focus on high-level thinking and guiding other agents. Not
> implementing things yourself nor bloating your context. I want following things
>
> 1) Top-to-down library and interface redesign.
>
> 2) Interface is consistent in all languages (C++, MATLAB and Python, like how Casadi is doing)
>
> 3) I want device selection like Pytorch so you create the DTWC environment then you set the device
> somehow. So it is like device=cpu, device=gpu, device=hpc. The cpu and gpu are local, and hpc is
> the SLURM interface we have. So it should use the credientials to connect HPC in the ".env" file.
> If they are not there then it should give an error that the connection is not established for the
> reason (no password -> then tell user how to put their things to .env, or wrong password etc.
> then tell user, so informative message then close). Maybe we could have some lazy loading so that
> if it is hpc then it doesn't load the data, or if the data is too big then it uses some mmap or
> something else. Meticulously decide these important design questions.
>
> 4) Automated compilation for executables and mex files, python wheels for all platforms. I think
> we can consider uploading the pypi when we are hundred percent sure our software is working. So we
> will release as DTWC++ 2.0.
>
> 5) The code is cross-platform, works on all supported platforms (windows, macos (both intel and
> amd), Linux (ubuntu)).
>
> 6) All parallelisation etc. things work out of the box. I don't want it to cannot activate
> parallelisation due to missing oneTBB etc. then fall back to sequential. Otherwise I would be
> happier probably for using std algorithms but this was the issue. Maybe we could make user-facing
> test interface like. dtwc.test.parallelisation() so this tries how many cores and how we can use
> it. Same for GPU testing. Once these functions are called it can just report back how many gpu
> what it is using etc.
>
> 7) See literature and other abilities we can add.
>
> 8) Zero overhead abstraction. So if we have the ability to choose L1 and L2 norms or inject
> another cost function. These should have nearly zero cost, you could in C++ especially inject
> things with compile time. And in other languages you could compile multiple options then the main
> function can select so the selection is not on the hotpath. Or anything that doesn't sacrifice
> speed. I want this library to be the fastest available DTWC++ library for large data etc. Or maybe
> multi-dim DTWC++ etc.
>
> 9) See my other attempts of writing my own solver, it doesn't work nicely but we could I think can
> improve. See also my work in UNIMODULAR.md where I believe this problem is almost unimodular, so
> in MIP programming I believe we could solve this very easily with a much more clever branching
> rather than leaving the solver to take the branches. And having a large MILP is impossible when
> you have lots of time series. LP would be more feasible and probably reducing this to a network
> problem and then solving somehow to global optimality would be amazing. Maybe you could throw
> another Claude Fable with max effort to investigate the math in the UNIMODULAR.md and maybe come
> up with a better solver also using my previous attempts. It would be nice to have something nice.
>
> 10) Once everything is there, we should update the documentation website.
>
> 11) Please think deeply and also remind me if I forgot anything. Like maybe you could write huge
> CUDA kernels other things or improve some algorithms to make this library EVEN FASTER. You could
> use some profiler some other thing see cache hit etc. You are free to change data types, how to
> hold data, how to do things. As long as this library is very fast, accurate, and portable.

## Standing rules that follow from the charter

- The orchestrating session designs, maps and reviews; implementation is delegated to subagents.
- Context is a budget. Read `MAP.md`, not the tree; read your wave's card in `PLAN.md`, not the
  whole plan; open a deep-dive report only for the section a task cites.
- A break needs a solid, written reason (`design.md` §2). Additive first.
- Tag, PyPI upload, ARC submission, `git push` and history rewrites are Volkan's actions, never an
  agent's.
