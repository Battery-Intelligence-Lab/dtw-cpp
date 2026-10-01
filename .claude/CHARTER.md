# DTWC++ 2.0 — Charter

Volkan's instructions, verbatim. Agents do not edit the quoted text; new instructions are appended
with a date. The target they lead to is in `MAP.md` and `DECISIONS.md`; how it is scheduled is in
`PLAN.md`.

## 2026-10-01

On whether GCC's FMA contraction should be turned off (at x86-64-v3 the SIMD lanes and per-pair kernels differ
in the last bits):

> We don't need bit-by-bit equivalence between compilers. So they could differ minimally like 1e-9 epsilon or
> something. However, this shouldn't change the clustering results. So as long as we have robust clustering
> accuracy, and only lose an epsilon level of accuracy then go for the speed of course.

On Parquet in the bindings:

> Btw, I think Python and MATLAB shouldn't deliver with their own parquet reader, is it too large library to bind?
> Because if we don't put this inside compiled bit, we could add into Python dependencies. So they could use the
> installed version so our python library would be somewhat more lightweight. Or would it increase the complexity.
> Think about it

> Great. If you consider other libraries we use for csv reading etc. Maybe it would be nicer to seperate some of the
> things so we can deliver lighter libraries for matlab and python. So our python and matlab things come with the
> essentials not libraries we use to read things. So we shouldn't just be like the bloated software who brings extra
> copies of the same things.

Asked whether the wheel and the MEX should keep HiGHS (75–78 % of each):

> There is highspy for python, isn't there also a matlab interface?

> Do you think it makes sense to remove it from python (so use highspy) and keep it in matlab?

On the build files:

> Also can you use CMakelists for folders like mip folder has one. And for algorithms folder you could have
> another one so you don't have to spell everything inside the cmakelists outside. So a nice quality improvement
> pass is needed here.

## 2026-09-30 — continue

> Please continue but don't call fable, for delegating simpler tasks use Sonnet 5.5 xhigh

Later the same day, on keeping `Nb` as `size_t` in the MIP backends to protect overflow guards:

> It is fine, we will never cluster as many as Nb^2 more than int64

> You are overthiking about simple things. Just keep the code simple, we think about it when it overflows

Later the same day, on build targets:

> We don't have to compile for each and every computer. We can put a border somewhere that is not too new.

> So for CUDA we can put some limits. Let's put the limit to A30, when it was relesed and same year ones

Answers of the same evening (to the orchestrator's questions):

- May W11c, W12a and W6f delete the tests the permission system refused? "Yes, all three".
- CPU floor for release archives and wheels: "x86-64-v3 (Recommended)".
- The ARC scripts:

> See this https://arc-user-guide.readthedocs.io/en/latest/_sources/arc-systems.rst.txt   available gpu and cpu
> hardware in slurm. We could definitely   device=hpc  and  gpu_device=...  name here. So that we could deploy the
> most specialised code if checking and automatically deploying the most high performance code was not available.
> If it is, then we could just internally detect it.

Later that evening, on the test deletions:

> Yes, please delete the trivial tests, we don't need to write tests just for writing tests.

## 2026-09-29 — continue

> Please go through what was done and try to continue. Don't forget our principles of no unnecessary
> abstraction. Clean code, maintainable code, is better than pages of abstractions. Also make sure the code
> works fast and generates decent assembly like SIMD where needed.

Later the same day (this overrides the 09-27 "use Fable max as advisor"):

> Did you use any fable on this treat? Please do not use it. Also stop now and write a handoff

## 2026-09-27 — bloat pass

> I was working on this on another computer, but my limit there finishes do I don't know where we are.
> But can you see the current situation and continue. Use Fable max for discussion. Do not create
> unnecessary things, extra defensive design, unnecessary abstractions etc. For example previously
> there was a integer checker was checking integers all the time, But we already know that you won't
> have more time series than the upper limit of the largest integer on the system. So you don't have
> to panick if you had a 32 bit integer and trillions of time series. Just take largest integer at
> compile time, default to 64 bit, and go on. I don't need lots of unnecessary garbage. I want a
> decent, maintainable, and correctly working library that can use CPU, GPU, HPC. So go through the
> library, re-evaluate the decisions, remove the bloat, have a nice interface for users, discuss with
> Fable max.

Later the same day:

> Okay use parallel agents and workflows to continue design

> use Fable max as advisor

## 2026-09-23 — YAGNI pass

> Recently I have found out that the plan was adding unnecessary complications to our library for
> example having complicate integer checking for overflow where we definitely cannot store more time
> series than the largest integer on the system, like in 32-bit integer system it is nearly impossible
> to store more than 2 billion time series right? and we have no issues especially for the 64-bit
> systems. Therefore, I told the previous agent to select just the largest integer and only check once
> while data is being loaded which is not even necessary. The person dealing with billions of time
> series would know what to do right? Then the agent changed it a bit, now I would like you to go over
> the plan once again for YAGNI principle. We don't need lots of useless abstractions, only the ones
> that keep the user interface stable, and keep the performance like SIMD-creation memory management
> etc. first class. Some helper functions for repeated functions. Then use a library where a
> compatible library with a compatible license (with BSD-3) exists and does the job portably. We don't
> need to rediscover the wheel. We need to write a decent library with top speed and compatible with
> CPU/GPUs/HPC. I really like
>
> device=cpu,   device=hpc,  device=gpu. type of setting at the beginning. So we set the problem
> nicely then it is decided what to do with the data, how to load it etc.
>
> I am just skeptical about file reading that is why I kept Arma for robust csv reading etc. Let's see
> the latest commits and the plan and also discuss Fable so we make a decent work.

Later the same day, on the proposed decisions:

> We previously tried highway it wasn't worth the effort.
> macOS wheel thing is which year?
> Eigen removal need to dig a little bit in why it w/o eigen is a bit slower all the time? Maybe there is
> something we can do.
> output names as it is fine.
> Let's continue

## 2026-09-22 — fragments

The full messages of that day were not recorded; these are the fragments the day's records quoted —
rows A-10 and D-22 of the plan draft left uncommitted on 2026-09-22, and row D-19 of `e784e5c` — copied
here so they outlive those rows.

> number of series should be checked only once when loading the data ... don't try to safeguard
> every little bit of the detail.

> the default types should be defined for machine like 64 bit or 32 bit ... once types are fixed then
> assume there is no overflow here. Less is more.

> not having bit-by-bit identicality is fine as long as it is correct

Recorded as paraphrase only (`DECISIONS.md` rule 18, `PLAN.md` D-18): avoid copyleft; prefer a mature
portable library over an in-tree rewrite.

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
- Context is a budget. Read `MAP.md`, not the tree; read your step in `PLAN.md` and the audit row it
  names, not the whole plan.
- A break needs a solid, written reason (`DECISIONS.md` §2). Additive first.
- Tag, PyPI upload, ARC submission, `git push` and history rewrites are Volkan's actions, never an
  agent's.
