""" nox session definition """

import nox

locations = "easyuq", "noxfile.py"


# https://cjolowicz.github.io/posts/hypermodern-python-02-testing/
# https://cjolowicz.github.io/posts/hypermodern-python-03-linting/


@nox.session(python=None)
def black(session):
    """runs black to do some automatic code automatisation"""
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@nox.session(python=None)
def lint(session):
    """runs the linter to inspect the code"""
    args = session.posargs or locations
    session.install(
        "flake8",
        # "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
    )
    session.run("flake8", *args)
