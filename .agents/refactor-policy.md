# Backwards-compatibility policy

- **Backwards compatibility is mandatory** for all public interfaces and supported behavior; changes must be additive and non-breaking by default.
- Internal interfaces may be refactored freely when all in-repository consumers are updated in the same change, provided no public behavior or interface is broken.
- Public breaking changes require explicit user approval as a scoped exception, a clear migration path, and prominent flags in code review, changelogs, and delivery reports; reviewers must reject unapproved breaks.
