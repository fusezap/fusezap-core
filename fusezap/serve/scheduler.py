"""Continuous batching scheduler (SPEC §11.3): admits waiting requests under a token budget, mixes prefill chunks and
decode tokens in one step, and preempts (recompute) when the paged KV cache runs out of blocks.

Policy (first come, first served, like vLLM's default):
  1. Running sequences that are past their prompt each get one decode token, oldest first. When a sequence needs a new
     block and none is free, the newest running sequence is preempted: its blocks are freed and it goes back to the
     waiting queue with everything it generated so far kept (recompute: its KV is rebuilt when it is readmitted).
  2. Running sequences still in their prompt get the next prompt chunk (chunked prefill), oldest first, under the
     prefill budget; a sequence that cannot get blocks preempts newer ones, or takes what fits.
  3. Waiting sequences are admitted in arrival order while the token budget, max_num_seqs and the free blocks allow.
     The queue head is never skipped, so a large request cannot be starved by later small ones; a sequence preempted in
     this step is not readmitted in the same step.
Since the oldest sequence never loses its blocks to a newer one, it always progresses, and every request finishes in a
bounded number of steps (each request must fit the pool alone: add() rejects prompt + max_tokens > capacity).

Token kinds: prompt positions are computed by prefill chunks; generated positions are computed one per step by decode
rows, also when they are recomputed after a preemption (teacher forced: the known token is fed, nothing is sampled).
So every position is always computed by the same kind of step, which keeps outputs bit-identical to running alone
(see fusezap/serve/engine.py).
"""
from __future__ import annotations
import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence as Seq
from fusezap.serve.kv_cache import PagedKVCache

class SeqStatus(enum.Enum):
  WAITING = "waiting"
  RUNNING = "running"
  FINISHED = "finished"

@dataclass(eq=False)
class Sequence:
  seq_id: int
  prompt: list[int]
  params: Any                      # engine.SamplingParams
  arrival: int
  output: list[int] = field(default_factory=list)
  num_computed: int = 0            # positions whose k/v are in the cache
  status: SeqStatus = SeqStatus.WAITING
  finish_reason: str | None = None
  sampler: Any = None
  cons: Any = None                 # constrain.Matcher: the grammar state of a constrained request (None: unconstrained)
  num_preemptions: int = 0
  preempted_at: int = -1           # scheduler step of the last preemption
  first_step: int = -1             # step that first scheduled it, and the step it finished (for starvation checks)
  finish_step: int = -1
  @property
  def tokens(self) -> list[int]: return self.prompt + self.output
  @property
  def num_tokens(self) -> int: return len(self.prompt) + len(self.output)
  @property
  def in_prompt(self) -> bool: return self.num_computed < len(self.prompt)

@dataclass
class SchedulerConfig:
  max_num_seqs: int = 32           # running sequences (= decode rows per step)
  max_prefill_tokens: int = 256    # prompt tokens per step (= prefill rows per step)
  max_num_batched_tokens: int | None = None   # decode + prefill tokens per step (default: the sum of the two above)
  max_prefill_seqs: int | None = None         # distinct sequences in one prefill chunk set (default max_num_seqs)
  mixed_prefill_tokens: int | None = None     # prompt tokens per step while sequences decode (None: max_prefill_tokens);
                                              # LLMEngine sets it for mixed forwards, or from FZ_DECODE_PREFILL_CAP
  prefill_every: int = 1                      # while sequences decode, prompt chunks only on every n-th step (decode
                                              # priority; FZ_PREFILL_EVERY); steps with nothing decoding always prefill
  prefill_first: int = 0                      # prefill-first (exact mode; FZ_PREFILL_FIRST): while prompt work is
                                              # pending, up to this many consecutive steps run the prompt chunks alone
                                              # and the decoding sequences wait (0: off). See Scheduler.schedule.
  prefill_buckets: tuple[int, ...] | None = None   # the prefill forwards' row counts (zero-padding sizing; None: off)
  split_rows: int = 256                       # zero-padding sizing: a forward's fixed cost in rows (FZ_PREFILL_SPLIT_ROWS)
  prefill_cost: dict[int, float] | None = None  # cost sizing (NEURON, I77): per prefill bucket the time of one forward (ms);
                                              # set, _fit covers the pending prompt tokens by the cheapest bucket sequence
  @property
  def token_budget(self) -> int: return self.max_num_batched_tokens or (self.max_num_seqs + self.max_prefill_tokens)

@dataclass
class SchedulerOutput:
  decode: list[Sequence] = field(default_factory=list)                   # one token each, at position num_computed
  prefill: list[tuple[Sequence, int, int]] = field(default_factory=list)  # (seq, start position, n tokens)
  preempted: list[Sequence] = field(default_factory=list)
  @property
  def num_tokens(self) -> int: return len(self.decode) + sum(n for _, _, n in self.prefill)
  @property
  def empty(self) -> bool: return not self.decode and not self.prefill

class Scheduler:
  def __init__(self, cfg: SchedulerConfig, cache: PagedKVCache):
    self.cfg, self.cache = cfg, cache
    self.waiting: deque[Sequence] = deque()
    self.running: list[Sequence] = []
    self.steps = 0
    self.num_preemptions = 0
    self.pf_streak = 0             # consecutive prefill-first steps (decoders waiting)
    self.num_prefill_first = 0     # steps that ran prompt chunks while decoders waited
    self.protect: set[int] = set()  # ids of sequences that must not be preempted (a launched step's, tokens not read yet)
  @property
  def capacity(self) -> int: return min(self.cache.num_blocks * self.cache.bs, self.cache.max_seq_len)
  def has_unfinished(self) -> bool: return bool(self.waiting or self.running)

  def add(self, seq: Sequence, max_tokens: int) -> None:
    need = len(seq.prompt) + max_tokens
    if not seq.prompt: raise ValueError("empty prompt")
    if need > self.capacity:
      raise ValueError(f"prompt ({len(seq.prompt)}) + max_tokens ({max_tokens}) = {need} exceeds the KV cache capacity of {self.capacity} tokens")
    seq.status = SeqStatus.WAITING
    self._enqueue(seq)

  def _enqueue(self, seq: Sequence) -> None:
    """Waiting queue in arrival order (a preempted sequence goes back ahead of everything that arrived after it)."""
    i = len(self.waiting)
    while i > 0 and self.waiting[i - 1].arrival > seq.arrival: i -= 1
    self.waiting.insert(i, seq)

  def finish(self, seq: Sequence, reason: str) -> None:
    seq.status, seq.finish_reason, seq.finish_step = SeqStatus.FINISHED, reason, self.steps
    self.cache.free(seq.seq_id)
    if seq in self.running: self.running.remove(seq)
    if seq in self.waiting: self.waiting.remove(seq)

  def preempt(self, seq: Sequence) -> None:
    self.cache.free(seq.seq_id)
    self.running.remove(seq)
    seq.num_computed, seq.status, seq.preempted_at = 0, SeqStatus.WAITING, self.steps
    seq.num_preemptions += 1
    self.num_preemptions += 1
    self._enqueue(seq)

  def _make_room(self, seq: Sequence, n_tokens: int, out: SchedulerOutput) -> bool:
    """Preempt the newest running sequences (newer than seq and not scheduled in this step) until seq can hold n_tokens;
    False if it still cannot."""
    busy = {id(s) for s in out.decode} | {id(s) for s, _, _ in out.prefill} | self.protect
    while not self.cache.can_ensure(seq.seq_id, n_tokens):
      newer = [s for s in self.running if s.arrival > seq.arrival and id(s) not in busy]
      if not newer: return False
      victim = max(newer, key=lambda s: s.arrival)
      self.preempt(victim)
      out.preempted.append(victim)
    return True

  def _prompt_work(self) -> bool:
    """Prompt tokens are waiting to be computed: a running sequence in its prompt, or an admissible waiting one."""
    return any(s.in_prompt for s in self.running) or bool(
      self.waiting and len(self.running) < self.cfg.max_num_seqs and self.waiting[0].preempted_at != self.steps)

  def prefill_first_ok(self) -> bool:
    """The next step will be prefill-first (prompt chunks alone, decoders waiting): enabled, within its bound, sequences
    decoding and prompt work pending."""
    cfg = self.cfg
    return bool(cfg.prefill_first and self.pf_streak < cfg.prefill_first and any(not s.in_prompt for s in self.running)
                and self._prompt_work())

  def schedule_prefill(self, protect: Seq[Sequence] = ()) -> SchedulerOutput | None:
    """A prefill-first step scheduled while the previous step's tokens are not read yet (LLMEngine's prefill pipeline):
    prompt chunks only, never the sequences in `protect` (they are not preempted either). None when prefill-first does
    not apply (off, its bound reached, or no prompt work) or nothing could be scheduled: then the caller reads the
    previous step's tokens and schedules normally."""
    cfg = self.cfg
    if not (cfg.prefill_first and self.pf_streak < cfg.prefill_first): return None
    self.steps += 1
    if not self._prompt_work(): return None
    out = SchedulerOutput()
    self.protect = {id(s) for s in protect}
    try: self._prefill(out, cfg.max_prefill_tokens)
    finally: self.protect = set()
    if not out.prefill: return None
    if any(not s.in_prompt for s in self.running): self.pf_streak += 1; self.num_prefill_first += 1
    return out

  def _fit(self, pbudget: int) -> int:
    """Zero-padding prefill sizing: a step whose prompt rows would not fill a prefill bucket is padded up to the next
    one (2560 rows run a 4096-row forward). When the prompt work ready now is below pbudget, run the largest bucket it
    fills and leave the rest for the next step, if that costs fewer rows (plus split_rows for the extra forward's fixed
    cost) than the padding. Every prompt position is computed by the same prefill kernels whatever the chunking and the
    bucket (packing- and chunk-invariant prefill), so this changes no token."""
    if self.cfg.prefill_cost: return self._fit_cost(pbudget)
    bs = [b for b in (self.cfg.prefill_buckets or ()) if b <= pbudget]
    if len(bs) < 2: return pbudget
    room, avail = self.cfg.max_num_seqs - len(self.running), 0
    for s in self.running:
      if s.in_prompt: avail += len(s.prompt) - s.num_computed
    for s in self.waiting:
      if room <= 0 or avail >= pbudget: break
      avail, room = avail + len(s.prompt), room - 1
    if avail >= pbudget: return pbudget
    hi = next((b for b in bs if b >= avail), pbudget)
    lo = max((b for b in bs if b <= avail), default=None)
    if lo is None or lo == avail or lo == hi: return pbudget
    rem = next(b for b in bs if b >= avail - lo)
    return lo if lo + rem + self.cfg.split_rows < hi else pbudget

  def _pending_prompt_tokens(self, cap: int) -> int:
    """Prompt tokens that could be prefilled now (running sequences' rest, then waiting ones while max_num_seqs has room),
    counted up to cap."""
    room, avail = self.cfg.max_num_seqs - len(self.running), 0
    for s in self.running:
      if s.in_prompt: avail += len(s.prompt) - s.num_computed
    for s in self.waiting:
      if room <= 0 or avail >= cap: break
      if s.preempted_at == self.steps: break
      avail, room = avail + len(s.prompt), room - 1
    return min(avail, cap)

  def _head_prompt_tokens(self) -> int:
    """The rest of the oldest prompt the next prefill starts with: a running sequence's (oldest first), else the first
    waiting one's when max_num_seqs has room; 0 when none."""
    for s in self.running:
      if s.in_prompt: return len(s.prompt) - s.num_computed
    if len(self.running) < self.cfg.max_num_seqs:
      for s in self.waiting:
        if s.preempted_at == self.steps: break
        return len(s.prompt)
    return 0

  def _fit_cost(self, pbudget: int) -> int:
    """Cost prefill sizing (cfg.prefill_cost: ms per forward of each bucket; NEURON, LLMEngine sets it when there are
    several FZ_NEURON_PREFILL_SIZES): the rest of the oldest pending prompt is covered by the sequence of forwards with
    the least estimated time, a forward of b rows costing cost[b] however many of its rows are filled, and this step runs
    the first one of it: its row budget is returned (other prompts fill the rows it leaves). E.g. 128 / 512 rows at
    53 / 195 ms: a 200-token prompt runs as two 128-row forwards (106 ms) instead of one 512-row one, 400 or more as 512.
    Per prompt, not the whole burst (I78): four 128-token prompts as one 512-row forward cost less in all (195 < 212 ms)
    but every one of them waits the whole 195 ms, where one by one the first is done in 53 (c4 TTFT p50 145 -> 204 ms).
    Scheduling only: which forward computes a prompt position changes no token (packing- and chunk-invariant prefill)."""
    cost = self.cfg.prefill_cost
    bs = sorted(b for b in cost if b <= pbudget and (not self.cfg.prefill_buckets or b in self.cfg.prefill_buckets))
    if len(bs) < 2: return pbudget
    head = min(self._head_prompt_tokens(), 8 * bs[-1])   # beyond a few of the largest forwards the first choice no longer changes
    if head <= bs[0]: return bs[0]
    memo: dict[int, float] = {}
    def cover(t: int) -> float:   # the least time to prefill t tokens
      if t <= 0: return 0.0
      if t not in memo: memo[t] = min(cost[b] + cover(t - b) for b in bs)
      return memo[t]
    return min(bs, key=lambda b: (cost[b] + cover(head - b), -b))   # ties: the bigger forward (fewer steps)

  def schedule(self) -> SchedulerOutput:
    """One step. Prefill-first (cfg.prefill_first > 0, exact mode): in exact mode a step with prompt chunks and decode
    rows runs two forwards, so while a burst of prompts is pending the decode forward (~6.7 ms for 7B) sits on the
    critical path of every step; a prefill-first step runs only the prompt chunks. At most prefill_first such steps in a
    row, then a step with the decode rows (and prompt chunks): a steady stream of arrivals cannot starve the decoders.
    Scheduling only: every position is still computed by its own kind of forward, so tokens do not change."""
    self.steps += 1
    pf_first = self.prefill_first_ok()
    cfg, out = self.cfg, SchedulerOutput()
    budget = cfg.token_budget
    self.running.sort(key=lambda s: s.arrival)
    if pf_first:
      self._prefill(out, cfg.max_prefill_tokens)
      if out.prefill:
        self.pf_streak += 1
        self.num_prefill_first += 1
        return out
    # 1. decode: one token per running sequence past its prompt, oldest first
    for seq in list(self.running):
      if seq not in self.running or seq.in_prompt: continue
      if budget <= 0: break
      if not self._make_room(seq, seq.num_computed + 1, out):
        self.preempt(seq)   # nothing newer to take blocks from: wait for older sequences to finish
        out.preempted.append(seq)
        continue
      self.cache.ensure(seq.seq_id, seq.num_computed + 1)
      out.decode.append(seq)
      budget -= 1
    if out.decode: self.pf_streak = 0
    # 2. prompt chunks of running sequences, oldest first
    pbudget = min(cfg.max_prefill_tokens, budget)
    if out.decode and cfg.mixed_prefill_tokens: pbudget = min(pbudget, cfg.mixed_prefill_tokens)   # keep a decoding step short
    if out.decode and cfg.prefill_every > 1 and self.steps % cfg.prefill_every: pbudget = 0
    self._prefill(out, pbudget)
    return out

  def _prefill(self, out: SchedulerOutput, pbudget: int) -> None:
    """Phases 2 and 3: prompt chunks of running sequences, then admissions, under pbudget prompt tokens."""
    cfg = self.cfg
    pseqs = cfg.max_prefill_seqs or cfg.max_num_seqs
    if pbudget > 0: pbudget = self._fit(pbudget)
    for seq in list(self.running):
      if seq not in self.running or not seq.in_prompt: continue
      if pbudget <= 0 or len(out.prefill) >= pseqs: break
      n = min(len(seq.prompt) - seq.num_computed, pbudget)
      if not self._make_room(seq, seq.num_computed + n, out):
        n = min(n, (len(self.cache.tables.get(seq.seq_id, ())) + self.cache.num_free_blocks) * self.cache.bs - seq.num_computed)
        if n <= 0: continue
      self.cache.ensure(seq.seq_id, seq.num_computed + n)
      out.prefill.append((seq, seq.num_computed, n))
      pbudget -= n
    # 3. admit waiting sequences in arrival order
    while self.waiting and pbudget > 0 and len(self.running) < cfg.max_num_seqs and len(out.prefill) < pseqs:
      seq = self.waiting[0]
      if seq.preempted_at == self.steps: break
      n = min(len(seq.prompt), pbudget)
      if not self.cache.can_ensure(seq.seq_id, n): break
      self.waiting.popleft()
      seq.status = SeqStatus.RUNNING
      if seq.first_step < 0: seq.first_step = self.steps
      self.running.append(seq)
      self.cache.ensure(seq.seq_id, n)
      out.prefill.append((seq, 0, n))
      pbudget -= n

class AdmitGate:
  """Gather a burst of short prompts into one prefill forward (I76; the engine loops of fusezap/serve/api.py ask it before
  each step). A closed-loop client (Invarix's c4) sends its next request right after the previous one ends, so when k
  sequences finish in one step, k prompts arrive a few ms apart (the server parses and submits them one by one). Without
  a gate the first one's prefill starts alone and the others wait a whole forward (NEURON 7B TP=2: a 128-row prefill is
  ~87 ms whether it holds one 40-token prompt or three). hold() is the seconds to wait before the next step: > 0 only while
    - FZ_ADMIT_WAIT_MS > 0 (`wait_ms`; default 0, off: I77 run 1's NEURON result was mixed; a CUDA prefill costs a few ms);
    - no sequence is decoding (a hold never stalls a decoder: TPOT is unchanged);
    - the pending prompt tokens leave room in the prefill forward (max_prefill_tokens) and in max_num_seqs, so prompts
      that fill a forward alone (Invarix's 128-token prompts on NEURON) never wait;
    - replacements are owed: more sequences finished within the last `recent_ms` (FZ_ADMIT_RECENT_MS, 250) than requests
      were added since (a lone client, c1, owes nothing once its next request is in);
  and at most `wait_ms` after the first of the waiting prompts was added. Scheduling only: tokens are unchanged."""
  def __init__(self, sch: "Scheduler", wait_ms: float, recent_ms: float = 250.0, clock=None):
    import time
    self.sch, self.wait, self.recent = sch, wait_ms / 1e3, recent_ms / 1e3
    self.clock = clock or time.perf_counter
    self.owed, self.t_fin, self.t_open = 0, -1e18, None
    self.holds = 0   # steps that waited (stats)
  def finished(self, n: int) -> None:
    """n sequences finished in the step that just ended."""
    if n <= 0: return
    now = self.clock()
    if now - self.t_fin > self.recent: self.owed = 0
    self.owed, self.t_fin = self.owed + n, now
  def added(self) -> None:
    """A request was added to the engine."""
    self.owed = max(0, self.owed - 1)
    if self.t_open is None: self.t_open = self.clock()
  def hold(self) -> float:
    left = self._left()
    if left <= 0: self.t_open = None   # the step goes now; the next add opens a new window
    return max(0.0, left)
  def _left(self) -> float:
    sch, now = self.sch, self.clock()
    if self.wait <= 0 or self.owed <= 0 or self.t_open is None or now - self.t_fin > self.recent: return 0.0
    if any(not s.in_prompt for s in sch.running): return 0.0
    rows = sum(len(s.prompt) - s.num_computed for s in sch.running if s.in_prompt) + sum(len(s.prompt) for s in sch.waiting)
    if rows == 0 or rows >= sch.cfg.max_prefill_tokens or len(sch.running) + len(sch.waiting) >= sch.cfg.max_num_seqs: return 0.0
    return self.t_open + self.wait - now
