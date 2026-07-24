# Scenario Prompt Conditions

`be_helpful_peer_reputation_pressure/` and
`no_guardrails_peer_reputation_pressure/` intentionally contain byte-for-byte
identical culprit-agent pressure prompts in all three scenarios:

- `scheduling_incident`
- `coding_incident`
- `financial_incident`

The experimental difference between these two conditions is on the
`PersonalAssistant` side, not in the culprit prompt.

`be_helpful_peer_reputation_pressure` uses the standard reporter-side files:

- `scenarios/personalassistant_system.txt`
- `scenarios/personalassistant_final_report.txt`
- `scenarios/shared_context.txt`

`no_guardrails_peer_reputation_pressure` uses the stripped reporter-side files:

- `scenarios/no_guardrails_personalassistant_system.txt`
- `scenarios/no_guardrails_personalassistant_final_report.txt`
- `scenarios/no_guardrails_shared_context.txt`

This keeps the pressure applied by the culprit agent fixed while testing
whether reporter-side instructions change responsibility attribution.
