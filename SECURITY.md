# Security Policy

Do not report real API keys in public issues. Revoke an exposed credential at
its provider first, then contact the repository maintainers privately.

The project treats model-generated HTML, Python, JavaScript, equations, object
IDs, and file names as untrusted input. Changes that weaken path confinement,
network blocking, import allowlists, expression restrictions, or deterministic
trace replay require focused security tests.

Only `.env.template` belongs in Git. A local `.env`, screening checkpoints,
raw model responses, request logs, and provider usage records must remain local.
