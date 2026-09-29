# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through the repository host's security
advisory feature. If that is unavailable, contact the maintainers through the
private contact method published with the release. Do not include credentials,
private model outputs, endpoint URLs containing secrets, or exploit details in
a public issue.

Include the affected version, reproduction steps, likely impact, and any safe
mitigation you have identified. The maintainers will acknowledge the report,
triage it, and coordinate disclosure after a fix is available.

## Scope

Security reports may cover credential leakage, unsafe path handling, malicious
dataset or model-output parsing, unintended network access, container escape,
or result-integrity failures that can silently misattribute benchmark scores.
Ordinary benchmark disagreements and model-quality findings should use the
normal issue tracker.

Never commit API keys. Supply credentials through environment variables or an
untracked `.env` file, inspect run metadata before sharing it, and run external
model servers in an appropriately isolated environment.
