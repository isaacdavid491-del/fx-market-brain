# How to test this against real prices

Everything measured so far used a made-up random walk. It cannot tell you
whether the strategy works. To get a real answer the code needs real NASDAQ
bars, which means an OANDA token and a machine that can reach OANDA.

There are two ways. The first needs nothing installed.

---

## Option A: use your Render deployment (no install)

This repository already deploys to Render, and the deployed service can fetch
its own data.

**1. Get an OANDA practice token.**

- Go to <https://www.oanda.com> and open a **practice** (demo) account. It is
  free and involves no real money.
- Sign in, then find *Manage API Access* (sometimes under *My Account* or
  *My Services*).
- Generate a token and copy it. It looks like two long strings joined by a
  dash.

**2. Put the token into Render.**

- Open <https://dashboard.render.com> and click your `fx-market-brain` service.
- Go to **Environment** in the left sidebar.
- Check that `OANDA_TOKEN` is listed. Click to edit it, paste your token, save.
- Render redeploys automatically. Wait for it to say **Live**.

**3. Let it collect data.** On the first start it downloads history, which
takes a few minutes. Check it is working by opening:

```
https://YOUR-SERVICE.onrender.com/api/ict/health
```

Look at `bars_stored` for `NAS100_USD`. It should be in the tens of thousands.
If it is 0, the token is not working; see Troubleshooting below.

**4. Run the test.** Open this in your browser:

```
https://YOUR-SERVICE.onrender.com/api/ict/validate?days=60&step_minutes=15
```

It takes several minutes. Leave the tab open.

**5. Read the answer.** Find the `status` field near the top:

| `status` | What it means |
|---|---|
| `real` | Real bars were replayed. The numbers mean something. |
| `synthetic` | The token is not working. The numbers mean nothing. |
| `no_data` | Nothing was tested. |

Copy the whole `text` field and paste it back into the chat.

> Render's free tier sleeps when idle and has limited memory. If the page
> times out, try `days=30&step_minutes=30` first.

---

## Option B: run it on your own computer

More setup, but faster and more reliable.

### Step 1 — install Python

**Windows:** download from <https://www.python.org/downloads/>, run the
installer, and **tick "Add python.exe to PATH"** on the first screen.

**Mac:** open Terminal and run `python3 --version`. If it prints a version of
3.11 or higher you already have it. Otherwise install from the same link.

### Step 2 — open a terminal

**Windows:** press Start, type `powershell`, open Windows PowerShell.

**Mac:** press Cmd+Space, type `terminal`, press Enter.

### Step 3 — download the code

```bash
git clone https://github.com/isaacdavid491-del/fx-market-brain.git
cd fx-market-brain
git checkout claude/nasdaq-ict-agent-farm-c3hjjz
```

No git? Download the ZIP from the branch page on GitHub, unzip it, then `cd`
into the unzipped folder.

### Step 4 — install the dependencies

```bash
pip install -r requirements-dev.txt
```

If `pip` is not found, use `python -m pip install -r requirements-dev.txt`
(on Mac, `python3 -m pip`).

### Step 5 — check it works before involving the broker

```bash
python -m pytest
```

You should see roughly 181 tests pass. If they do, the code is healthy and
anything that goes wrong next is about data or the token.

### Step 6 — set your token

**Windows PowerShell:**
```powershell
$env:OANDA_TOKEN = "paste-your-token-here"
```

**Mac / Linux:**
```bash
export OANDA_TOKEN="paste-your-token-here"
```

This lasts only for that terminal window. Open a new one and you must set it
again.

### Step 7 — run the test

```bash
python -m backend.cli validate --days 60
```

It downloads about sixty days of one-minute bars, then replays the agents
across three exit policies. Expect ten to twenty minutes.

### Step 8 — read the top line

```
==============================================================================
  REAL MARKET DATA
==============================================================================
```

That banner is the only thing that makes the rest worth reading. If it says
`SYNTHETIC DATA` or `NO USABLE DATA`, nothing was measured.

Copy everything it printed and paste it into the chat.

---

## Troubleshooting

**"SYNTHETIC DATA" with a token set.** The token was not picked up. Check for
typos and stray quotes, and confirm you set it in the same terminal window you
ran the command in.

**"NO USABLE DATA - NOTHING WAS TESTED".** The token reached nothing. Usually
one of: the token is for a live account while the code points at the practice
API, the token was revoked, or the network blocked the request.

**"seeding failed: 401".** The token is wrong or expired. Generate a new one.

**"seeding failed: 403".** Something between you and OANDA is blocking it,
often a corporate network or VPN. Try a home connection.

**`python` is not recognised (Windows).** Python was installed without being
added to PATH. Re-run the installer, choose Modify, and tick the PATH option.

**It seems stuck.** It is not. Sixty days of one-minute bars is about sixty
thousand rows and the agents evaluate thousands of decision points. Add
`--step 15` to make it roughly three times faster with fewer decisions.

---

## What a real result will look like

Whatever comes back, these are the things worth reading:

- **Trades.** Around one or two a day would match the intended selectivity.
  Far more means the gates are too loose; far fewer means too tight.
- **Win rate against break-even rate.** The farm needs roughly a 25% hit rate
  to cover costs at its usual reward-to-risk. Below that it loses regardless
  of how good the analysis looks.
- **The paired comparison.** Whether the exit policies keep the sign they had
  on synthetic data.
- **Sample size.** Sixty days gives about sixty trades, which the report will
  tell you is not enough to trust a small effect. It is enough to see a big
  one, in either direction.

A flat or negative result on real data is a genuine finding, and more useful
than anything produced so far.
