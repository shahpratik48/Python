# Demo Speech — Delivery Metrics & Productivity Dashboard
**Audience: management · Target: 6–7 minutes (~850 words at a natural pace)**

Timings are cumulative. Bracketed lines are delivery notes, not spoken.

---

### 0:00 — The problem *(45s)*

Good morning. I want to show you two things we've built to answer a question we've never been able to
answer objectively: **are we getting faster, and how do we know?**

Until now, delivery metrics were manual — someone pulling numbers out of GitLab into a spreadsheet, a
few times a year, and not consistently between pods. So we couldn't compare pods fairly, we couldn't
see a trend, and by the time a problem surfaced it was already a quarter old.

These two projects fix that. Everything I'm about to show you is automated, runs every weekday, and
needs nobody maintaining a spreadsheet.

---

### 0:45 — What we measure *(60s)*

*[Have the four-panel chart on screen.]*

We measure four things — two about **volume**, two about **speed**.

On volume: **Issue Throughput** — how many work items we complete in a month. And **Merged MRs per
Author** — how many code changes each engineer actually lands. That second one is deliberately a
*ratio*, not a raw count, because a raw count simply rises when we add people. The ratio tells us
about each engineer's flow.

On speed: **Issue Cycle Time** — once work starts, how many working days until it's done. And **Lead
Time** — from the first line of code written to the moment it's merged.

For the two volume metrics, higher is better. For the two speed metrics, lower is better. The system
handles that inversion for you, so a *falling* cycle time always shows as a *positive* number.

---

### 1:45 — Project 5: the reports *(75s)*

*[Open a monthly report, then the daily one.]*

The first project produces the reports. Every pod gets four files — a **monthly** report covering the
current month and the previous three, and a **daily** report, each in HTML and Excel, same content,
same styling.

The monthly report answers *how did we do*. Four metrics per month, broken down per author and per
repository, with anything outside our service levels flagged — issues taking more than five business
days, merge requests sitting longer than seventy-two hours.

The daily report is the one I'd actually watch. It runs at 6pm each weekday and answers a different
question: **what needs attention today.** It shows two groups — work that has already breached, and
more usefully, work that is *still open* and will breach before the day ends unless someone finishes
it.

That's the difference between a report and an early warning. A breach already recorded is history. An
item about to breach can still be saved — and 6pm still leaves the evening to act.

---

### 3:00 — Project 6: reading the dashboard *(105s)*

*[Switch to the dashboard. Point at the baseline line as you speak.]*

The second project is the trend view — the same four metrics, but over twenty-one months, so you see
direction rather than a snapshot.

There's one idea that unlocks the whole chart.

**November 2025 is our baseline.** Everything is measured against that month. Before it, the line is
*dotted* — that's the run-up, history. From it, the line is *solid* — that's where we are now.

The dashed line is the **aspiration** — our target. It starts exactly on top of the actual at the
baseline, because ambition starts from where you actually were, and it ramps up twenty percent over
fourteen months. So the gap between the solid line and the dashed line **is** the gap between where
we are and where we said we'd be.

The percentage at the top of each panel is simply: *how far is the latest point from the baseline
point.* And the headline at the very top — Overall Productivity Gain — is the plain average of those
four percentages. Nothing more sophisticated than that; you can check it by eye from the four panels.

Every point on the chart is labelled with its value, so you can read any number straight off without
asking anyone to look it up.

---

### 4:45 — Two views *(35s)*

There are two versions of every chart. One shows each month individually — that's the detail. The
other shows rolling three-month averages — November to January, December to February, and so on.

I'd use the three-month view for any trend conversation, because a single month for a single pod
swings hard on one late ticket.

---

### 5:20 — One honest note *(30s)*

One request when you read these: **look at the four panels, not just the headline.** The overall
number is an unweighted average, so a single strong metric can carry it while others go backwards.
The detail is right there on the same page.

---

### 5:50 — Scale and close *(50s)*

Finally — this scales. Everything is driven by a configuration file. Adding a pod is a few lines of
config; we added **DS Data** this week without touching a line of code. The org hierarchy — division,
subdivision, stream, crew, pod — works the same way, with multi-select filters, so this extends
beyond our crew whenever you want it to.

To summarise: we've replaced a manual, occasional, inconsistent process with something **automatic,
daily, and comparable across pods**. It tells us where we stand against a baseline we agreed, and
every evening it tells us what still needs attention.

Happy to take questions.

---

## If you're running short — cut these first

1. The **Two views** section (4:45) — mention it in one line while showing the chart instead.
2. The per-author and per-repository detail in the monthly report.

## If you have spare time — add these

- Show the **Pod filter** and switch between pods live. It lands the configurability point better than
  describing it.
- Open the **Excel** version to make the point that it's the same content, for anyone who wants to
  pivot it themselves.

## Likely questions, and short answers

| Question | Answer |
|---|---|
| *Why November 2025?* | It's the agreed reference month. Everything is measured against it, so the numbers stay comparable over time rather than shifting with a rolling window. |
| *Where does the 20% come from?* | It's our stated aspiration over fourteen months. It's a configuration setting — if the target changes, the line changes. |
| *Can this cover other crews?* | Yes. The hierarchy and pods are configuration, not code. |
| *Is this measuring individuals?* | No. It's per pod. Author breakdown exists in the detail sheet for flow analysis, not performance rating — worth saying that plainly if it comes up. |
| *One number looks extreme — why?* | Percentages are measured against a single baseline month, so an unusually quiet or busy baseline month exaggerates them. Check the monthly detail before drawing a conclusion. |
| *How much effort to maintain?* | None routinely. It runs on a schedule; adding a pod is a config change. |
