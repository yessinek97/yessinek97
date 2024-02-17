# Contribute to biondeep-IG

To contribute to the [`biondeep-IG`](https://gitlab.com/instadeep/biondeep-ig) repository, please first install [the Conda based setup](installation.md#conda-based-setup-for-developers).

>🚨 **Please make sure you [install the pre-commit](installation.md#conda-based-setup-for-developers) before committing any changes.**

## Issue

Please [create a new/separate](https://gitlab.com/instadeep/biondeep-ig/-/issues/new#) issue if

- There is a bug or problem in the existing code.
- Some code need to be refactored/moved for a better code structure.
- There are redundant code to be simplified.
- A new feature needs to be added.
- Some new ideas or approaches we could test.

For each issue, please

- **Include only one issue/problem.** For instance, if we want to change code structure, we **should
  not** remove redundant code in the same issue/MR. So that each issue/MR is sweet and short.
- Describe what should be done and how we should handle the issue.
- Estimate how much time we need.
- Assign a milestone and a person if possible.
- Use the corresponding issue template on gitlab if it exists (e.g. bugs, data, general-purpose)

## Merge Request

To handle a raised issue, please

1. Go to the issue, and click `Create merge request` to automatically create a new branch based on the latest `main` branch. The created branch will automatically follow the repository's naming convention.

    >🚨 **Please avoid creating branches manually.**

2. Move to the **new branch** in your local development environment.
    `git checkout new_branch`
3. Work on your contributions **locally**.
4. **Commit frequently** with meaningful commit messages and **push frequently** so that you can make sure no tests are failed.
5. **Fix tests ASAP** when they are failed during the development.
6. **Synchronise** with `main` branch frequently.

    ```bash
    git  fetch
    git rebase origin/main
    ```

    >🚨 **Resolve any conflicts immediately**

7. When the MR is ready, please request **code review** and address all discussions.

## Code Review

When doing code review, please check

- If the MR resolves and **only resolves** the corresponding issue. For instance, if the issue is to
  change code structure, unless necessary, removing redundant code should not be in the same MR.
- If the added/modified code are test covered as much as possible. Code needs to be modular and
  therefore easier to add unit test.
- If added/modified code has been well-documented:
  - **Statical Typing**: Function arguments/return should have correct type hinting.
  - **Docstring**: Function arguments/return/raised errors are documented in the docstring of
    [google style](https://google.github.io/styleguide/pyguide.html).
  - **Tensor Shape Notation**: Tensor shapes are documented in the comments for better understanding
    of the data flow.
  - **Meaningful naming**: Function/variable names are meaningful and consistent. Classes should be
    CamelCase such as `TransformerEncoder`, functions and variables should be lowercase with
    underscores such as `get_embedding` and `global_embedding`.
- If the added/modified code are simple, without over-engineering. **We do not recommend over
  engineering the code, means that sometime some redundancy is preferred for code simplicity.**

In addition, we have the following preferences

- We should use the `Path` from `pathlib` instead of `os`.
- We should not use `assert` as [it can be disabled](https://docs.python.org/3/using/cmdline.html#cmdoption-O).

At last, we use code reviews to

- learn from each other,
- transfer knowledge about the code base,
- improve the code quality in the long run,
- catch bugs.

Therefore, please be nice and objective and do not take comments personally.
