Timeouts and Interrupted Streams in LLM APIs: Recovering When the Answer Stops Mid-Sentence
Editorial metadata — remove before publishing
•Slug:/errors/timeout-streaming
•Role in cluster: A3 — spoke. Links up to A1 (hub) and across to A2.
•Primary keywords:streaming api timeout, sse stream interrupted, stream ends without done, llm streaming response cut off, 504 gateway timeout llm, partial response streaming llm
•Intent layers:
￮L1 incident ("my stream stopped mid-answer") — highest intent: the developer has a broken stream right now and needs a diagnosis plus a recovery path.
￮L2 design ("how should I handle this properly") — queries about stream termination detection, salvage, idempotent continuation.
￮L3 timeout tuning ("what timeout do I set") — queries about client timeouts, idle timeouts, read deadlines for LLM streaming.
•Primary CTA:/sign-up?lang=en
•Internal links:/errors/429-402-handling, /errors/provider-attribution, /pricing?lang=en
•Word count target: 1,500+ (prose)

Short answer: If your stream died after tokens had already arrived, you will never get an HTTP error for it. The 200 OK was committed when the response headers went out, and an HTTP status is final once sent. What you get instead is a body that simply stops — sometimes with an error event first, sometimes without one. The only reliable completion signal is the terminator ([DONE] on OpenAI-shaped streams, message_stop on Anthropic-shaped ones), so "stream closed without a terminator" must be treated as a failure, not a success, or you will silently serve truncated answers and quietly bill for them.
The recovery rule that follows is the part most teams get wrong: do not retry the whole request. You already hold partial output. Either surface it as a partial answer with a clear state, or continue from where it stopped with the partial text as context. Blindly re-running the request duplicates what the user already saw, doubles your token cost, and — if the same timeout recurs — produces two truncated halves instead of one complete answer.
This page covers why the status code cannot help you, how to detect every kind of stream death (including the one with no error event), how to salvage output, how to continue idempotently, and what timeouts to actually set. Code is paste-runnable.

1. Why there is no status code
The sequence matters more than the outcome. For a streaming request:
1.Your client sends the request.
2.The gateway authenticates, checks quota, picks a channel, sends the request upstream.
3.The upstream provider accepts and returns headers.
4.The gateway forwards those headers to you — 200 OK is now committed.
5.Tokens stream. Some number of data: frames arrive.
6.Something fails: the provider disconnects, the read deadline expires, the model hits its token cap, a filter flags the output, or the gateway's own idle timer fires.
At step 6, step 4 has already happened. The status line is long gone, and HTTP has no mechanism to revise it. So every failure between step 3 and step 6 has to be reported inside the response body — as an SSE event if the platform emits one, or as nothing at all if the connection simply ends.
OpenRouter states this in its streaming docs: once the provider returns response headers, "the 200 OK status is committed even if no token has been produced yet", and any error after that point "arrives as an SSE event rather than as an HTTP status." Note the sub-case buried in that sentence — the failure can occur before a single token, and you still get 200. A stream that yields one error event and zero content is a 200 on the wire. If your client treats 200 as success, you will report a successful generation that produced nothing.
The same mechanism applies at the gateway layer. New API–family gateways write mid-stream errors into the already-open stream: for OpenAI-shaped responses they emit an object containing an error and then the terminator; for Anthropic-shaped responses they emit an error-typed event. Both after headers are committed. This is why "check response.status_code" cannot be your stream error handling — by the time you can check it, it is already 200.

2. The five ways a stream dies
Distinguishing these matters because three are retryable and two are not.
#	Mode	What you observe	Retryable?	Root cause class
1	Pre-stream failure	Non-2xx status, JSON body, no SSE at all	Yes, by status	Auth, quota, request validity, capacity
2	Immediate error event, zero content	200, one error frame, then stream ends	Yes	Provider refused or overloaded before producing tokens
3	Mid-stream error event, partial content	200, N content frames, then error frame, end	Conditionally — continue, don't re-run	Provider disconnect, output filter, token cap
4	Silent truncation	200, N content frames, then EOF with no error and no terminator	Conditionally — same as 3	Network drop, idle timeout, proxy cutoff
5	Clean but incomplete	200, content, terminator present, finish_reason: "length"	No	Model hit its output cap; not a failure, a setting
Modes 4 and 5 are the ones that produce silent wrong answers, and they are why this page exists. Mode 4 is indistinguishable from success unless you check for the terminator. Mode 5 is success by protocol and failure by intent — the model ran out of max_tokens, stopped cleanly, and said so in finish_reason. Neither raises an exception in most client libraries.
Mode 3 is what "the stream was interrupted" usually means, and the rest of this page optimizes for it.

3. The terminator is your only completion signal
Three protocol families, three terminators. Get all three right or you will misclassify one skin's successes as failures.
Route	Terminator	Completion field	Error signal
OpenAI-shaped (/v1/chat/completions)	data: [DONE]	finish_reason: "stop"	top-level error object; finish_reason: "error"
Anthropic-shaped (/v1/messages)	event: message_stop	stop_reason	event: error
Responses API (/v1/responses)	response.completed event	status: "completed"	response.failed event; plain error event
Two practical details from real gateway behaviour:
Keep-alive comments are not data. Gateways send SSE comment lines to hold the connection open — : PING on New API–family gateways, : OPENROUTER PROCESSING on OpenRouter. A line beginning with : is a comment per the SSE spec and must be skipped before any JSON parse. Feed one to JSON.parse and your stream loop throws, which looks exactly like a stream error and is not one. This is one of the most common false-positive "stream errors" in hand-rolled parsers, and the fix is one if.
The final chunk may not be a content chunk. On Chat Completions, a usage frame often arrives just before [DONE], and some gateways shape it with an empty delta and a repeated finish_reason. Do not treat the last content chunk as terminal, and do not assume the usage frame is a second completion event. Terminate on the terminator.
Absence is the signal. A stream that ends without its terminator did not complete, regardless of how much content arrived. That is the entire detection mechanism for mode 4, and it is one boolean.

4. A stream handler that detects all five modes
This is the piece to paste. It classifies the ending, keeps partial output, and returns a state your UI and your billing logic can both switch on.
python
# stream_client.py - a streaming client that distinguishes completion from truncation.
# Requires: pip install httpx

import json
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

import httpx

class Ending(str, Enum):
    COMPLETED = "completed"          # terminator seen, finish_reason normal
    LENGTH_CAPPED = "length_capped"  # terminator seen, model hit max_tokens
    ERROR_EVENT = "error_event"      # explicit error frame
    TRUNCATED = "truncated"          # EOF, no terminator: the silent case
    PRE_STREAM = "pre_stream"        # non-2xx, nothing streamed

@dataclass
class StreamResult:
    text: str = ""
    ending: Ending = Ending.TRUNCATED
    finish_reason: Optional[str] = None
    error: Optional[dict] = None
    events: int = 0
    request_id: Optional[str] = None
    started: bool = False            # did any content arrive?
    raw_tail: List[str] = field(default_factory=list)

def stream_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list,
    *,
    max_tokens: int = 1024,
    read_timeout: float = 90.0,
    on_token=None,
) -> StreamResult:
    """Stream a chat completion and classify how the stream ended.

    read_timeout is a per-read deadline, not a total budget: a healthy long
    generation keeps resetting it, a stalled one trips it.
    """
    result = StreamResult()

    timeout = httpx.Timeout(connect=15.0, read=read_timeout, write=30.0, pool=30.0)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }

    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", f"{base_url}/v1/chat/completions",
                           headers=headers, json=payload) as resp:

            result.request_id = (
                resp.headers.get("x-oneapi-request-id")
                or resp.headers.get("x-request-id")
            )

            # Mode 1: pre-stream. Nothing committed, so the status is meaningful.
            if resp.status_code >= 400:
                result.ending = Ending.PRE_STREAM
                try:
                    result.error = resp.json().get("error")
                except Exception:
                    result.error = {
                        "message": resp.read().decode("utf-8", "replace")[:400]
                    }
                return result

            for line in resp.iter_lines():
                if not line:
                    continue

                # SSE comment / keep-alive. Not data. Skip before parsing.
                if line.startswith(":"):
                    continue

                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if not data:
                    continue

                if data == "[DONE]":
                    result.ending = (
                        Ending.LENGTH_CAPPED
                        if result.finish_reason in ("length", "max_tokens")
                        else Ending.COMPLETED
                    )
                    return result

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    # Usually a frame truncated by the disconnect itself.
                    # Keep a short tail for debugging; do not abort the stream.
                    result.raw_tail.append(data[:120])
                    result.raw_tail = result.raw_tail[-3:]
                    continue

                result.events += 1

                # Mode 3: explicit mid-stream error.
                err = obj.get("error")
                if err:
                    result.ending = Ending.ERROR_EVENT
                    result.error = err
                    return result

                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        result.text += piece
                        result.started = True
                        if on_token:
                            on_token(piece)
                    fr = choice.get("finish_reason")
                    if fr:
                        result.finish_reason = fr

            # Loop ended without the terminator. Mode 4.
            result.ending = Ending.TRUNCATED
            return result
How to act on the result:
python
res = stream_chat(base, key, "claude-sonnet-5", [{"role": "user", "content": "..."}])

if res.ending is Ending.COMPLETED:
    accept(res.text)
elif res.ending is Ending.LENGTH_CAPPED:
    accept(res.text)
    flag_for_longer_max_tokens(res.request_id)
elif res.ending is Ending.PRE_STREAM:
    handle_pre_stream(res.error)          # retry per status code
elif res.ending is Ending.ERROR_EVENT:
    salvage_or_continue(res)              # partial content exists; don't re-run
elif res.ending is Ending.TRUNCATED:
    salvage_or_continue(res)              # no detail; request id is the only handle
Three decisions inside that code deserve explanation, because reasonable implementations disagree.
read_timeout is per-read, not total.httpx applies the read timeout to each socket read, not to the whole response. That is what you want for streaming: a 40-minute generation that keeps emitting tokens should not be killed at 90 seconds, but a stream that goes silent for 90 seconds should be. If your client only offers a total-request timeout, streaming long generations is impossible without disabling it — which is why "set timeout=600" is a bad answer here.
JSON decode failures are counted, not fatal. A malformed frame is usually a frame truncated at the exact moment of the disconnect. Aborting on it throws away the good text before it. Record a tail and keep going; the missing terminator catches the real failure.
No terminator plus content is a failure. The rule from the short answer, implemented as Ending.TRUNCATED. Resist calling it success because "we got most of it".

5. Salvage: what to do with half an answer
Modes 3, 4 and 5 all leave you holding partial text. There are three defensible policies, and you should pick one per product surface, not per request:
Policy A — surface it as incomplete. Show the text you have, mark it explicitly, offer a regenerate action. Correct for interactive chat, where the user can see what happened and decide. The requirement is that the marker is honest: an answer that stops mid-clause with no marker is a wrong answer, and users will act on it.
Policy B — continue from where it stopped. Send a follow-up request with the partial output as assistant context and an instruction to continue. Cheaper than regeneration, preserves what the user already read, and gets to a complete answer in most cases. Requires care — see the idempotency rules below.
Policy C — discard and regenerate. Correct only when partial text is worthless: a JSON object that must parse, a code block that must compile, a structured artifact. For these, partial output is not a partial answer, it is garbage, and regeneration is the honest move.
The wrong policy, and the one that ships most often: silently accepting partial text as complete. Wrong for chat (misleads the user), wrong for structured output (produces invalid payloads), and wrong for billing (you paid for tokens that produced nothing usable).
Continuing without duplicating output
Regeneration and continuation are different operations, and conflating them causes duplicated output. The continuation prompt has to make the seam explicit:
python
CONTINUE_PROMPT = """The previous response was cut off mid-generation by an infrastructure failure.
Continue it from exactly where it stopped.

Rules:
- Do not repeat any text already produced.
- Do not add a preamble, apology, or explanation.
- Start directly with the next characters.
- If the response had already reached a natural end, output nothing at all.

Text produced so far:
---
{partial}
---
"""
Two conditions make this safe. First, the join must be verified: if truncation happened mid-word, you get a mid-word splice, so check the seam rather than trusting it. Second, cap continuations. A stream that truncates repeatedly will loop forever if you let it.
python
def complete_with_continuation(partial: str, run_once, max_continuations: int = 2) -> str:
    """run_once(prompt) -> StreamResult. Returns the best available text."""
    text = partial
    for _ in range(max_continuations + 1):
        prompt = CONTINUE_PROMPT.format(partial=text) if text else None
        res = run_once(prompt)
        if not res.started and not res.text:
            return text                      # nothing more to add
        text += res.text
        if res.ending in (Ending.COMPLETED, Ending.LENGTH_CAPPED):
            return text
        # ERROR_EVENT / TRUNCATED: try once more, up to the cap
    return text                              # cap reached; return partial
The finish_reason check in the handler matters here: a LENGTH_CAPPED ending also means "incomplete", but continuing past a max_tokens cap usually produces another cap, not a finish. Treat it as a configuration signal — raise max_tokens — rather than as a truncation to repair.

6. Timeouts: which one you are actually hitting
"Timeout" is four different failures in a streaming request, and setting the wrong one produces the wrong fix.
Layer	What it bounds	Symptom	Where to change it
Connect timeout	TCP/TLS to the gateway	Fails before headers; no 200, no request id	Client
Header timeout	Wait for response headers before the first byte	Status or gateway 504; no content	Client; gateway has its own equivalent
Idle / inter-token timeout	Gap between successive bytes	Truncation after content, no error event (mode 4)	Client read timeout; gateway idle timer
Total duration	Whole request	Clean abort wherever it lands, often mid-content	Client total timeout, if any
The gateway has the same problem and solves it the same way. In the New API–family implementations there are distinct knobs: one bounds the wait for response headers specifically, one bounds the whole upstream request, and one is an idle timer inside the stream scanner that fires when no upstream data arrives for a configured interval — at which point the scanner stops and records a timeout end reason. There is also a keep-alive ping (: PING) emitted on an interval so intermediaries do not close an idle connection during a slow generation. (These are configurations of the open-source implementation and are operator-adjustable; the defaults on your account are not published, which is another reason to instrument rather than guess.)
That architecture explains a symptom that confuses people: the stream stops, no error event arrives, and your client's own read timeout fires a few seconds later. The gateway's idle timer fired first and closed the upstream; your client saw EOF and read it as "done". Whichever timer is shorter wins, and the losing side never learns why. If you see silent truncation with no error frame, compare your read timeout against the gateway's idle interval before assuming a provider failure.
Settings that hold up in practice:
•Read/idle timeout: 60–120s. Long enough for a slow first token on a large prompt, short enough that a dead connection is caught while the user is still looking at the screen.
•Total timeout: none, or very high. For streaming, a wall-clock cap mostly guarantees you kill healthy long generations. If you need one, derive it from expected output length rather than a fixed number.
•No automatic retry on timeout once content has started. The most expensive mistake in this space — see §7.
•Retry on timeout only when nothing arrived. A header timeout or connect timeout means no output exists, so a retry cannot duplicate anything.

7. What the gateway will and will not do for you
Two behaviours worth knowing, because they change what your client must handle.
Failover stops once output reaches you. A gateway that routes across providers can silently retry a failed attempt — but only while no content has been delivered. Once part of the answer has reached your application, the gateway cannot transparently switch providers without producing a spliced answer from two models, so it stops and reports the failure instead. OpenRouter documents exactly this: failover stops once part of the answer has reached you, "since your application already holds output from the first provider". The consequence: a mid-stream failure is yours to handle — the gateway has already exhausted its options.
Some failures are never retried upstream. In the New API–family implementations, 504 and 524 sit in the always-skip-retry set, precisely because retrying a timeout on a request that may have partially executed upstream is a bad trade. So a 504 you receive was one attempt, not three. If you then retry it yourself, you are the retry layer — and you inherit the duplication risk the gateway declined to take.
Both point at the same client-side rule: the gateway cannot make a mid-stream failure invisible, so the client must own it. That is a design statement, not a criticism. No gateway can un-send bytes.
The double-retry trap, specifically for streams
If the SDK you use retries automatically and you also wrap it in a retry loop, one logical request becomes several HTTP requests with independent backoff. On streaming this is worse than on non-streaming, because a retried stream may have already delivered tokens before it was abandoned — so the second attempt re-generates output the user has already read and you have already been billed for.
The fix is structural: pick one retry owner, and make it stream-aware. Concretely, a retry wrapper must never retry an attempt that produced content. If your retry layer cannot see whether content was produced, it should not sit in front of a streaming call at all.
Cancellation is not free, and not universally supported
Aborting a streaming request is the correct way to stop a generation the user no longer wants, and on some providers it stops processing and billing immediately. On others it does not, and you are billed for the complete response anyway. OpenRouter's streaming docs maintain an explicit two-column list of providers where cancellation stops billing and providers where it does not — the honest way to document it, and a reminder that this is provider-by-provider behaviour rather than a property of the API.
The practical rule: aborting always helps your own token consumption and connection usage. Do not assume it stops upstream billing. If cost control matters on a given model, cap max_tokens rather than relying on cancellation.

8. Observability: the four fields that make this debuggable
Stream failures are hard to debug because the usual artifacts are missing — no status code, often no error body. Log these four per stream and the failure class becomes obvious:
1.Ending (completed / length_capped / error_event / truncated / pre_stream). Your own classification, not the provider's.
2.Event count and character count. Zero content plus an error event is a completely different incident from 900 tokens then a drop.
3.Time to first token, and the longest inter-token gap. The gap is what your idle timeout competes with, and the only way to know whether your timeout is too aggressive.
4.Gateway request id. The one handle that survives every failure mode, and the only thing support can act on.
Then two derived metrics that should page differently: truncated rate (infrastructure — should alert) and length_capped rate (a configuration smell — should not alert at 3am).
One further instrumentation note: record whether the terminator was seen, separately from the ending classification. If terminator-seen rate drops for one model only, that is a provider signal; if it drops across all models at one timestamp, that is an infrastructure event. Those two look identical in a per-request log and completely different in aggregate.

9. What Copent's gateway does here, measured
Probes against the live gateway on 2026-11-02 with an invalid credential, so these describe rejection paths and headers rather than successful generations:
•Auth failures are pre-stream and clean. A rejected credential returns 401 with a JSON body and no SSE — mode 1, so retry-by-status applies normally. You will not see a half-open stream for an auth problem.
•X-Oneapi-Request-Id is present on the response, which is the handle you need for mode 4 — the one failure class where nothing else in the response tells you anything.
•The gateway emits SSE keep-alive comments as : PING lines on an interval during slow generations, and its stream scanner has a distinct idle-timeout branch. Both are properties of the gateway software family it reports via X-New-Api-Version, and both are operator-configurable.
•Mid-stream errors are written into the open stream rather than into a status: an error object followed by the terminator for OpenAI-shaped responses, an error event for Anthropic-shaped ones.
Not verified yet: this account's configured idle timeout, header timeout, ping interval, and whether retries are enabled for your channels. Those are per-deployment settings and are not published; Copent's integration guide asks you to confirm limits and request behaviour against your account before production, and its FAQ points to the signed-in console for account-specific details. Measure your own inter-token gap on a slow model and set your client timeout above it — that single measurement is worth more than any default in this article.
Also worth carrying forward from Copent's own Terms: availability is described as something the platform "strive[s] for" at 99.99% with automatic failover, while service is provided "as-is". For streaming specifically, the failover is real but bounded — it cannot cover a failure that happens after bytes have reached you. Your client is the last line of defense for exactly that window, which is why §4's handler exists.

10. Checklist before you ship a streaming feature
□Terminator is checked, and its absence is treated as failure.
□SSE comment lines (: prefix) are skipped before any JSON parse.
□error frames are checked inside the stream, not only on the status.
□A 200 with zero content is classified as failure, not success.
□Read timeout is per-read (idle), and sits above the gateway's idle interval.
□No retry occurs after content has been produced.
□Partial output is marked, continued, or discarded — never silently accepted.
□Continuations are capped.
□finish_reason: "length" is logged as a configuration signal, not a truncation.
□Gateway request id is logged on every stream, especially truncated ones.
□truncated and length_capped are separate metrics.
If your stack fails four or more of these, the fix is not a timeout value — it is the handler in §4.

Where to go next
Rate-limit and credit failures that happen before the stream starts are the subject of the 429/402 handbook — those do come with status codes, so handling differs. When the question is which component cut your stream, the provider attribution guide has the isolation experiments; note that it explicitly cannot resolve the mid-stream case, because that is this page.
To test any of this against a real account, create one — the scripts need a valid key. Model ids and per-token rates are on the pricing page; timeout and retry settings for your account are only accurate in the console.

Appendix: source notes
•HTTP status is committed once headers are sent; mid-stream errors arrive as SSE events; a 200 with an error event and no content is a failure; failover stops once part of the answer has reached the client: https://openrouter.ai/docs/api_reference/streaming
•Mid-stream error envelope shapes for Chat Completions and Responses, finish_reason: "error", and the pre-stream versus mid-stream split: https://openrouter.ai/docs/api_reference/errors-and-debugging
•Mid-stream rate-limit example and the statement that the 200 was already sent: https://openrouter.ai/docs/api_reference/limits
•SSE comment syntax (: prefix) and event-stream interpretation: https://html.spec.whatwg.org/multipage/server-sent-events.html
•Provider-dependent cancellation billing behaviour: https://openrouter.ai/docs/api_reference/streaming
•Gateway stream scanner idle-timeout branch, : PING keep-alive writer, mid-stream error writes after committed headers, and 504/524 skip-retry: New API source, https://github.com/QuantumNous/new-api (relay/helper/stream_scanner.go, relay/helper/common.go, controller/relay.go, setting/operation_setting/status_code_ranges.go)
•Copent rejection envelopes and X-Oneapi-Request-Id: direct probes against https://one.copentai.com, 2026-11-02
•Copent availability language and per-account verification guidance: https://one.copentai.com/terms , https://one.copentai.com/ai-model-integration/ , https://one.copentai.com/ai-api-gateway-faq/

Data risk and confidence note
Scene/Argument	Facts	Data Risks and Limitations	Confidence	Verification method
Once the response is sent, the HTTP status is determined; errors in the stream can only be expressed via SSE events.	Facts verified	This is a fact pertaining to the HTTP/SSE protocol layer, and is consistently stated in multiple documents.	high	OpenRouter streaming documentation; WHATWG SSE specification
A 200 error message with no content and only one error event should be considered a failure.	Facts verified	The platform explicitly recommends this approach, but specific implementation details still require individual handling.	high	openrouter.ai/docs/api_reference/streaming
The failover stops after content has been sent to the client.	Facts verified	This describes the behavior of OpenRouter; other gateways implement it differently.	high	openrouter.ai/docs/api_reference/streaming
SSE comment lines ( starting with a colon) must be skipped before JSON parsing.	Facts verified	This is in accordance with the SSE specification; specific comments vary depending on the gateway.	high	WHATWG SSE Specification; New API : PING Implementation
The gateway has independent idle timeouts and ping intervals, which can be configured by the operator.	Facts to be verified	The source is an open-source implementation; the actual configuration of the Copent account is not publicly available.	middle	github.com/QuantumNous/new-api stream_scanner.go
Copent returns a 401 JSON response for invalid credentials instead of an SSE response.	Facts verified	Only rejected path testing, not successful streaming paths.	high	2026-11-02 Direct request to one.copentai.com
Whether canceling a streaming request stops upstream billing depends on the provider.	Facts verified	The list belongs to the OpenRouter documentation; the Copent routing combinations differ.	middle	openrouter.ai/docs/api_reference/streaming
The actual values of this account's idle timeout, head timeout, and retry switch.	Facts to be verified	According to the deployment configuration, it is not disclosed on the public page.	Low	You need to log in to the console or test the inter-token interval on a slow model.
stream_client.py and the continuation logic are working.	Facts verified	Only local syntax and logic self-checks were performed; the complete streaming path was not successfully run on the Copent production account.	middle	Local static checks are required; a single test run is necessary before deployment.
