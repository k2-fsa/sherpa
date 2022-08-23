from sphinx.application import Sphinx
from docutils.parsers.rst import roles
from docutils import nodes
from docutils.parsers.rst.states import Inliner


def strike_role(
    role, rawtext, text, lineno, inliner: Inliner, options={}, content=[]
):
    your_css_strike_name = "strike"
    return (
        nodes.inline(rawtext, text, **dict(classes=[your_css_strike_name])),
        [],
    )


def setup(app: Sphinx):
    roles.register_canonical_role(
        "my-strike", strike_role
    )  # usage:  :my-strike:`content ...`
