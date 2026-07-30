# Outreach email templates

Four versions, one per target category (see `outreach/target-selection.md`
for how to pick a specific recipient). Each is 150–250 words, written in
first person as Kadir Eren Altıntaş. **`[bracketed fields]` must be
filled in with real, verified specifics about the actual recipient
before sending — never send with a placeholder still in it.** No
version claims an existing relationship, a prior conversation, or a
result that hasn't happened. None of these have been sent; this file is
template source only.

---

## 1. University systems/architecture professor

**Subject:** KV-cache memory research (simulation + RTL) — looking for FPGA/CXL access

Dear Professor [Last name],

I'm Kadir Eren Altıntaş, an independent researcher (42 İstanbul) working
on MEMBRANE, an open-source project on LLM KV-cache memory management:
mixed-precision tiering verified bit-exact against a production
quantizer, and an exact (non-approximate) sparse retrieval design,
evaluated via discrete-event simulation at 128K context × 512
concurrency, with a synthesizable FPGA datapath cosimulated (Verilator,
520,000 transactions, zero mismatches) against the same reference math.

I'm reaching out because [your group's specific published work on
memory systems / KV-cache / near-memory computing — cite the specific
paper] is directly relevant to what I'm trying to validate next: real
place-and-route and, ideally, real board measurement, since everything
hardware-adjacent in this project today is simulation or a synthesis
cell-count check, not a real result.

I'm not asking for a commitment — just whether your lab has FPGA
toolchain access (Vivado/Quartus + a board) that could accommodate a
short, scoped experiment (see `docs/phase8-hardware-validation-plan.md`
Level A/B for exactly what that would involve), possibly as a student
project if that's a better fit than your own time.

Repository: github.com/kadireren7/membrane · Paper draft:
`paper/main.md` · 25-second reproducible demo: `scripts/demo.sh`.

Happy to send more detail or adjust scope to whatever access is
realistic on your end.

Best regards,
Kadir Eren Altıntaş

---

## 2. FPGA / reconfigurable computing laboratory

**Subject:** Synthesizable, bit-exact-verified KV-quantization RTL — seeking board access

Hello [Lab name] team,

I'm Kadir Eren Altıntaş, sole author of MEMBRANE, an open-source LLM
KV-cache research project. The relevant piece for your group: a fully
synthesizable, purely-integer fixed-point Q8/Q4 quantization datapath
(no `real`/DPI anywhere), cosimulated in Verilator against a real CPU
reference for 520,000 transactions with zero mismatches, and confirmed
to elaborate cleanly under yosys 0.33. Real `synth_ecp5` cell counts
exist per module (the FP32 divider is the dominant cost, ~73,600
LUT-class cells, un-pipelined — a disclosed timing-closure risk, not
hidden). No Fmax, no place-and-route, and no board result exist yet —
this environment has no P&R tool or FPGA hardware.

Given [your lab's specific FPGA/board infrastructure or published
accelerator work], I wanted to ask whether a short, scoped
place-and-route attempt (and, if that closes timing, a loopback-DMA
bring-up) on hardware you already have access to would be of any
interest — either as a favor, a student project, or a collaboration if
the result is interesting either way.

The RTL is vendor-IP-free by design (interface skeletons only for
AXI4-Stream/AXI-Lite/DMA — see `hardware/README.md`), so it shouldn't
require redistributing anything proprietary.

Repository: github.com/kadireren7/membrane ·
`docs/phase8-hardware-validation-plan.md` has the full scoped plan.

Thank you for considering this — no obligation either way.

Kadir Eren Altıntaş

---

## 3. CXL / memory-systems research team

**Subject:** Simulated CXL near-memory KV-cache design — looking for a real CXL platform to check it against

Hello [Team/group name],

I'm Kadir Eren Altıntaş, working on MEMBRANE, an open-source project
modeling a near-memory/CXL appliance for LLM KV-cache memory: a
discrete-event simulator calibrated from real captured attention traces,
evaluated at 128K context × 512 concurrency (462/462 scenarios
complete), with every CXL link-latency/bandwidth figure drawn from the
CXL Consortium's own published specification as an explicit, cited
assumption — no real CXL hardware has been used anywhere in this
project, and I want to change that.

I came across [your team's specific CXL/memory-tiering publication or
platform] and thought the fit was close enough to ask directly: does
your team have access to a CXL Type-3 memory device, or an emulation/
prototyping platform, that could accommodate a small, scoped integration
test (see `docs/phase8-hardware-validation-plan.md` Level C)? I'm
specifically trying to find out whether the simulator's queueing/
contention model resembles real CXL device behavior at all — a question
I currently cannot answer myself.

This would be useful to me even as a negative result (i.e. if the
simulation turns out to be a poor match) — I'd rather know that than
not.

Repository: github.com/kadireren7/membrane · Paper: `paper/main.md`.

Open to whatever scope is realistic given your platform's constraints.

Kadir Eren Altıntaş

---

## 4. Company research or hardware-prototyping team

**Subject:** Open-source KV-cache/FPGA research project — requesting hardware access, not a sales conversation

Hello [Team name],

I'm Kadir Eren Altıntaş, an independent researcher and the sole author
of MEMBRANE (github.com/kadireren7/membrane), an open-source project on
LLM KV-cache memory: mixed-precision tiering bit-exact-verified against
a production quantizer, exact (non-approximate) sparse retrieval, and a
synthesizable FPGA quantization datapath cosimulated against the same
CPU math (520,000 transactions, zero mismatches). Everything is
published with sourced numbers and a claim-audit process
(`paper/claim-audit.md`) — I'm flagging that up front because this
email is a genuine research-access request, not a sales pitch.

I'm reaching out because [your team's specific relevant hardware
prototyping capability, product, or publication] looks like it could
help answer the one question this project can't answer on its own: does
this RTL actually synthesize, close timing, and run correctly on real
silicon? Right now every hardware number in this project is a
cosimulation or a synthesis cell-count, not a board result.

I'd like to ask whether your team could provide time-boxed or remote
access to an FPGA board + toolchain (or a CXL platform, if relevant) for
a scoped validation pass — full plan and pass/fail criteria in
`docs/phase8-hardware-validation-plan.md`. I'm not asking for funding,
IP, or exclusivity — just hardware access and, if useful, engineering
input on board-specific integration.

Happy to adjust scope to whatever's realistic.

Kadir Eren Altıntaş
