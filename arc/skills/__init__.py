"""Skills integration — ARC over the SKILL.md collection.

The Luo-Kai skill repo bundles ~9,786 ``SKILL.md`` files in the standard
Anthropic/Claude format (YAML frontmatter + markdown instructions). These are
*progressive-disclosure knowledge*, not executable Python: ARC discovers
them through a compact searchable index (``skills-index.json``) and loads a
single skill's full instructions on demand to steer the model.

Exposed as a small set of tools::

    skills.search(query)  ->  find matching skills (name + description)
    skills.list(category) ->  enumerate skills in a category / all
    skills.load(name)     ->  return one skill's full instructions as context

Because only the (small) index is kept hot and individual files are read from
disk on demand, the whole library adds negligible context while staying
searchable across the entire collection.
"""

from .index import SkillLibrary, SkillEntry
from .tools import register_skills_tools

__all__ = ["SkillLibrary", "SkillEntry", "register_skills_tools"]
