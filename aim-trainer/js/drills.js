/* Drill definitions. Each drill owns its own state, drawing and scoring. */
var AT = window.AT || {};

(function () {
  var TAU = Math.PI * 2;

  function median(arr) {
    if (!arr.length) return 0;
    var s = arr.slice().sort(function (a, b) { return a - b; });
    var m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  /* Place a target at a random bearing and distance from the current view. */
  function spawnAround(g, minFrac, maxFrac, radiusPx) {
    var minDim = Math.min(g.w, g.h);
    var angle = Math.random() * TAU;
    var dist = minDim * (minFrac + Math.random() * (maxFrac - minFrac));
    return {
      x: g.view.x + Math.cos(angle) * dist,
      y: g.view.y + Math.sin(angle) * dist * 0.72,
      r: radiusPx * g.targetScale
    };
  }

  function screenOf(g, t) {
    return { x: g.cx + (t.x - g.view.x), y: g.cy + (t.y - g.view.y) };
  }

  function onTarget(g, t) {
    var dx = t.x - g.view.x, dy = t.y - g.view.y;
    return Math.sqrt(dx * dx + dy * dy) <= t.r;
  }

  /* ---------- shared drawing ---------- */

  function drawTarget(ctx, x, y, r, opts) {
    opts = opts || {};
    var hot = opts.active !== false;
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, TAU);
    ctx.fillStyle = hot ? 'rgba(255,107,53,.22)' : 'rgba(120,140,165,.12)';
    ctx.fill();
    ctx.lineWidth = hot ? 2.5 : 1.5;
    ctx.strokeStyle = hot ? '#ff6b35' : 'rgba(139,155,176,.55)';
    ctx.stroke();

    if (opts.lit) {
      ctx.beginPath();
      ctx.arc(x, y, r * 0.34, 0, TAU);
      ctx.fillStyle = '#3ddc84';
      ctx.fill();
    } else if (hot) {
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, r * 0.14), 0, TAU);
      ctx.fillStyle = '#ffb03a';
      ctx.fill();
    }

    if (opts.decay !== undefined && opts.decay < 1) {
      ctx.beginPath();
      ctx.arc(x, y, r + 6, -Math.PI / 2, -Math.PI / 2 + TAU * opts.decay);
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(255,176,58,.75)';
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawOffscreenCue(g, ctx, t) {
    var p = screenOf(g, t);
    var m = 26;
    if (p.x > m && p.x < g.w - m && p.y > m && p.y < g.h - m) return;
    var dx = p.x - g.cx, dy = p.y - g.cy;
    var a = Math.atan2(dy, dx);
    var rad = Math.min(g.w, g.h) * 0.36;
    var x = g.cx + Math.cos(a) * rad;
    var y = g.cy + Math.sin(a) * rad;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(a);
    ctx.beginPath();
    ctx.moveTo(12, 0); ctx.lineTo(-6, -7); ctx.lineTo(-6, 7); ctx.closePath();
    ctx.fillStyle = 'rgba(255,107,53,.8)';
    ctx.fill();
    ctx.restore();
  }

  function drawGrid(g, ctx) {
    var step = 96;
    var ox = -(g.view.x % step), oy = -(g.view.y % step);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,.035)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (var x = ox - step; x <= g.w + step; x += step) {
      ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, g.h);
    }
    for (var y = oy - step; y <= g.h + step; y += step) {
      ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(g.w, Math.round(y) + 0.5);
    }
    ctx.stroke();
    ctx.restore();
  }

  function pct(n, d) { return d > 0 ? (n / d) * 100 : 0; }

  function signed(n) { return (n > 0 ? '+' : '') + n; }

  /* ================= DRILLS ================= */

  var list = [];

  /* ---- 1. Flick shots ---- */
  list.push({
    id: 'flick',
    name: 'Flick Shots',
    icon: 'flick',
    blurb: 'One target at a time. Swing onto it and fire.',
    trains: 'Raw target acquisition — the swing from where your crosshair is to where the enemy actually is. This is the single biggest source of lost gunfights on a touchscreen.',
    how: [
      'Drag anywhere on the screen to swing your view.',
      'Put the centre crosshair on the orange target.',
      'Tap FIRE. A new target spawns immediately.',
      'An arrow points the way when the target is off-screen.'
    ],
    mode: 'aim',
    duration: 45,
    primary: { label: 'median time to target', unit: 'ms', decimals: 0, lowerIsBetter: true },

    init: function (g) {
      g.state = {
        target: spawnAround(g, 0.18, 0.42, 34),
        spawnAt: g.t,
        path: 0,
        direct: 0,
        times: [],
        effs: [],
        shots: 0,
        hits: 0
      };
      g.state.direct = Math.hypot(g.state.target.x - g.view.x, g.state.target.y - g.view.y);
      g.hudSub = '0 hits';
    },

    update: function (g, dt) {
      g.state.path += g.viewDelta;
    },

    draw: function (g, ctx) {
      drawGrid(g, ctx);
      var p = screenOf(g, g.state.target);
      drawTarget(ctx, p.x, p.y, g.state.target.r, { lit: onTarget(g, g.state.target) });
      drawOffscreenCue(g, ctx, g.state.target);
    },

    onFire: function (g) {
      var s = g.state;
      s.shots++;
      if (onTarget(g, s.target)) {
        s.hits++;
        s.times.push((g.t - s.spawnAt) * 1000);
        s.effs.push(s.direct > 1 ? s.path / s.direct : 1);
        g.hit(screenOf(g, s.target));
        s.target = spawnAround(g, 0.18, 0.42, 34);
        s.spawnAt = g.t;
        s.path = 0;
        s.direct = Math.hypot(s.target.x - g.view.x, s.target.y - g.view.y);
        g.hudSub = s.hits + ' hits';
      } else {
        g.missFx();
      }
    },

    finish: function (g) {
      var s = g.state;
      return {
        primary: s.times.length ? median(s.times) : 0,
        counted: s.times.length >= 3,
        metrics: [
          { k: 'Hits', v: String(s.hits) },
          { k: 'Accuracy', v: Math.round(pct(s.hits, s.shots)) + '%' },
          { k: 'Fastest', v: s.times.length ? Math.round(Math.min.apply(null, s.times)) + 'ms' : '—' },
          { k: 'Path waste', v: s.effs.length ? signed(Math.round((median(s.effs) - 1) * 100)) + '%' : '—' }
        ],
        eff: s.effs.length ? median(s.effs) : 1,
        acc: pct(s.hits, s.shots)
      };
    },

    coach: function (r) {
      if (!r.counted) return 'Too few hits to score. Run it again and prioritise landing shots over speed.';
      if (r.eff > 1.75) return 'You are travelling ' + Math.round((r.eff - 1) * 100) + '% further than the straight line to the target, which is the classic overshoot-and-correct pattern. Drop your look sensitivity by about 15% and run this again.';
      if (r.eff < 1.12 && r.acc > 80 && r.primary > 900) return 'Your swings are efficient but slow — you are creeping onto targets. Raise sensitivity slightly and commit to the first swing.';
      if (r.acc < 60) return 'You are firing before the crosshair settles. Wait for the centre dot to turn green, then fire. Accuracy first, speed follows.';
      return 'Clean run. Your swing path is efficient and you are firing on settle. Push for faster acquisition without letting accuracy drop below 75%.';
    }
  });

  /* ---- 2. Tracking ---- */
  list.push({
    id: 'track',
    name: 'Tracking',
    icon: 'track',
    blurb: 'Keep the crosshair glued to a moving target.',
    trains: 'Holding aim on a strafing enemy. Most mobile players can flick but lose the target the instant it moves laterally.',
    how: [
      'The target drifts continuously. No firing required.',
      'Keep the centre crosshair inside the circle.',
      'It glows green whenever you are on target.',
      'Score is the share of the drill you spent on target.'
    ],
    mode: 'aim',
    duration: 30,
    primary: { label: 'time on target', unit: '%', decimals: 1, lowerIsBetter: false },

    init: function (g) {
      g.state = {
        target: { x: g.view.x + 120, y: g.view.y, r: 40 * g.targetScale },
        origin: { x: g.view.x, y: g.view.y },
        on: 0,
        streak: 0,
        bestStreak: 0,
        phase: [Math.random() * TAU, Math.random() * TAU, Math.random() * TAU, Math.random() * TAU]
      };
      g.hudSub = 'on target 0%';
    },

    update: function (g, dt) {
      var s = g.state;
      var t = g.t;
      var amp = Math.min(g.w, g.h) * 0.30;
      s.target.x = s.origin.x + Math.sin(t * 0.9 + s.phase[0]) * amp + Math.sin(t * 2.3 + s.phase[1]) * amp * 0.35;
      s.target.y = s.origin.y + Math.sin(t * 0.7 + s.phase[2]) * amp * 0.5 + Math.sin(t * 1.9 + s.phase[3]) * amp * 0.22;

      if (onTarget(g, s.target)) {
        s.on += dt;
        s.streak += dt;
        if (s.streak > s.bestStreak) s.bestStreak = s.streak;
      } else {
        s.streak = 0;
      }
      g.hudSub = 'on target ' + Math.round(pct(s.on, g.t)) + '%';
    },

    draw: function (g, ctx) {
      drawGrid(g, ctx);
      var s = g.state;
      var p = screenOf(g, s.target);
      drawTarget(ctx, p.x, p.y, s.target.r, { lit: onTarget(g, s.target) });
      drawOffscreenCue(g, ctx, s.target);
    },

    onFire: function () {},

    finish: function (g) {
      var s = g.state;
      var share = pct(s.on, g.elapsed);
      return {
        primary: share,
        counted: g.elapsed > 10,
        metrics: [
          { k: 'On target', v: share.toFixed(1) + '%' },
          { k: 'Longest hold', v: s.bestStreak.toFixed(1) + 's' },
          { k: 'Time on target', v: s.on.toFixed(1) + 's' },
          { k: 'Drill length', v: Math.round(g.elapsed) + 's' }
        ],
        streak: s.bestStreak
      };
    },

    coach: function (r) {
      if (r.primary < 35) return 'You are chasing the target rather than leading it. Watch where the circle is heading and move your thumb with it, not after it. Small continuous corrections beat big catch-up swings.';
      if (r.streak < 1.5) return 'Your longest unbroken hold was under one and a half seconds. That is the number to grow — a mobile gunfight is usually decided inside two seconds of sustained fire.';
      if (r.primary > 70) return 'Strong tracking. Try this again one notch above your usual sensitivity to build headroom, then drop back.';
      return 'Solid. Aim to push time on target past 70% before you raise the difficulty.';
    }
  });

  /* ---- 3. Target switching ---- */
  list.push({
    id: 'switch',
    name: 'Target Switching',
    icon: 'switch',
    blurb: 'Four targets. Kill the lit one, then the next.',
    trains: 'Re-acquiring after a kill. The gap between dropping one enemy and putting rounds on their teammate is where most multi-kills die.',
    how: [
      'Four circles are on the field. Only the orange one is live.',
      'Swing onto it and tap FIRE.',
      'The next target lights up instantly — keep the chain going.',
      'Score is your median time between kills.'
    ],
    mode: 'aim',
    duration: 45,
    primary: { label: 'median switch time', unit: 'ms', decimals: 0, lowerIsBetter: true },

    init: function (g) {
      var s = { targets: [], active: 0, since: g.t, times: [], shots: 0, hits: 0 };
      for (var i = 0; i < 4; i++) s.targets.push(spawnAround(g, 0.20, 0.46, 32));
      g.state = s;
      g.hudSub = '0 kills';
    },

    update: function () {},

    draw: function (g, ctx) {
      drawGrid(g, ctx);
      var s = g.state;
      for (var i = 0; i < s.targets.length; i++) {
        var t = s.targets[i];
        var p = screenOf(g, t);
        var isActive = i === s.active;
        drawTarget(ctx, p.x, p.y, t.r, {
          active: isActive,
          lit: isActive && onTarget(g, t)
        });
      }
      drawOffscreenCue(g, ctx, s.targets[s.active]);
    },

    onFire: function (g) {
      var s = g.state;
      s.shots++;
      var t = s.targets[s.active];
      if (onTarget(g, t)) {
        s.hits++;
        s.times.push((g.t - s.since) * 1000);
        g.hit(screenOf(g, t));
        s.targets[s.active] = spawnAround(g, 0.20, 0.46, 32);
        var next = s.active;
        while (next === s.active) next = Math.floor(Math.random() * s.targets.length);
        s.active = next;
        s.since = g.t;
        g.hudSub = s.hits + ' kills';
      } else {
        g.missFx();
      }
    },

    finish: function (g) {
      var s = g.state;
      return {
        primary: s.times.length ? median(s.times) : 0,
        counted: s.times.length >= 3,
        metrics: [
          { k: 'Kills', v: String(s.hits) },
          { k: 'Accuracy', v: Math.round(pct(s.hits, s.shots)) + '%' },
          { k: 'Fastest switch', v: s.times.length ? Math.round(Math.min.apply(null, s.times)) + 'ms' : '—' },
          { k: 'Chain rate', v: (s.hits / Math.max(1, g.elapsed) * 60).toFixed(0) + '/min' }
        ],
        acc: pct(s.hits, s.shots)
      };
    },

    coach: function (r) {
      if (!r.counted) return 'Not enough kills to score this run. Slow the swings down until you are landing them, then rebuild speed.';
      if (r.acc < 55) return 'More than four in ten shots missed. You are firing mid-swing. Let the swing stop, confirm the green dot, then fire.';
      if (r.primary < 700) return 'Excellent switch speed. Turn target size down to Small in Setup and run it again.';
      return 'Good chain. Work on cutting dead time right after the kill — the next target is already lit before your thumb moves.';
    }
  });

  /* ---- 4. Recoil control ---- */
  list.push({
    id: 'recoil',
    name: 'Recoil Control',
    icon: 'recoil',
    blurb: 'Hold fire and drag down to fight the climb.',
    trains: 'Pulling down through a full magazine. Sprays climb and drift; the players who win long fights are the ones countering it without thinking.',
    how: [
      'Hold the FIRE button down to spray. The view climbs and sways.',
      'Drag downward to counter the climb and keep the crosshair on the target.',
      'The magazine holds 30 rounds, then reloads automatically.',
      'Score is the share of your fired rounds that were on target.'
    ],
    mode: 'aim',
    duration: 40,
    primary: { label: 'rounds on target', unit: '%', decimals: 1, lowerIsBetter: false },

    init: function (g) {
      g.state = {
        target: { x: g.view.x, y: g.view.y - 40, r: 38 * g.targetScale },
        mag: 30,
        magSize: 30,
        reloading: 0,
        fireCd: 0,
        shots: 0,
        hits: 0,
        seed: Math.random() * 100,
        climb: 0
      };
      g.hudSub = 'mag 30/30';
    },

    update: function (g, dt) {
      var s = g.state;
      if (s.reloading > 0) {
        s.reloading -= dt;
        if (s.reloading <= 0) {
          s.mag = s.magSize;
          s.climb = 0;
          g.hudSub = 'mag ' + s.mag + '/' + s.magSize;
        } else {
          g.hudSub = 'reloading';
        }
        return;
      }

      if (!g.firing) {
        s.climb = Math.max(0, s.climb - dt * 3.2);
        return;
      }

      s.fireCd -= dt;
      if (s.fireCd <= 0 && s.mag > 0) {
        s.fireCd += 0.1;
        s.mag--;
        s.shots++;
        if (onTarget(g, s.target)) s.hits++;
        g.shotFx();
        g.hudSub = 'mag ' + s.mag + '/' + s.magSize;
        if (s.mag === 0) s.reloading = 1.4;
      }

      /* Recoil: strong early climb that eases off, plus horizontal sway. */
      s.climb = Math.min(1, s.climb + dt * 1.6);
      var vertical = 260 * (1 - s.climb * 0.55);
      var sway = Math.sin((g.t + s.seed) * 6.5) * 95 + Math.sin((g.t + s.seed) * 2.7) * 55;
      g.view.y -= vertical * dt;
      g.view.x += sway * dt * s.climb;
    },

    draw: function (g, ctx) {
      drawGrid(g, ctx);
      var s = g.state;
      var p = screenOf(g, s.target);
      drawTarget(ctx, p.x, p.y, s.target.r, { lit: onTarget(g, s.target) });
      drawOffscreenCue(g, ctx, s.target);

      /* magazine bar */
      var bw = 120, bh = 5;
      var bx = g.cx - bw / 2, by = g.h - 44;
      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,.12)';
      ctx.fillRect(bx, by, bw, bh);
      ctx.fillStyle = s.reloading > 0 ? '#8b9bb0' : '#ff6b35';
      var frac = s.reloading > 0 ? (1 - s.reloading / 1.4) : s.mag / s.magSize;
      ctx.fillRect(bx, by, bw * clamp(frac, 0, 1), bh);
      ctx.restore();
    },

    onFire: function () {},

    finish: function (g) {
      var s = g.state;
      var share = pct(s.hits, s.shots);
      return {
        primary: share,
        counted: s.shots >= 20,
        metrics: [
          { k: 'On target', v: share.toFixed(1) + '%' },
          { k: 'Rounds fired', v: String(s.shots) },
          { k: 'Rounds landed', v: String(s.hits) },
          { k: 'Mags emptied', v: String(Math.floor(s.shots / s.magSize)) }
        ]
      };
    },

    coach: function (r) {
      if (!r.counted) return 'You barely fired. Hold the FIRE button down — this drill is about sustained sprays, not taps.';
      if (r.primary < 40) return 'The climb is beating you. Start your downward drag at the same moment you start firing, not after you see the crosshair rise.';
      if (r.primary > 75) return 'Very strong recoil control. Switch to Small targets in Setup to keep this drill hard.';
      return 'Decent. The first eight rounds climb fastest — front-load your pull and ease off as the spray settles.';
    }
  });

  /* ---- 5. Precision taps ---- */
  list.push({
    id: 'taps',
    name: 'Precision Taps',
    icon: 'taps',
    blurb: 'Tap targets directly before they expire.',
    trains: 'Thumb precision and screen geography for tap-to-fire layouts. No crosshair — this is pure touch accuracy.',
    how: [
      'Targets appear anywhere on the screen and shrink as they expire.',
      'Tap each one before its ring runs out.',
      'There is no crosshair and no fire button in this drill.',
      'Score is targets destroyed per minute.'
    ],
    mode: 'tap',
    duration: 40,
    primary: { label: 'targets per minute', unit: '', decimals: 0, lowerIsBetter: false },

    init: function (g) {
      g.state = { targets: [], hits: 0, taps: 0, offsets: [], nextAt: 0, life: 1.35 };
      g.hudSub = '0 hits';
    },

    update: function (g, dt) {
      var s = g.state;
      s.nextAt -= dt;
      if (s.nextAt <= 0 && s.targets.length < 3) {
        s.nextAt = 0.55;
        var m = 52 * g.targetScale;
        s.targets.push({
          x: m + Math.random() * (g.w - m * 2),
          y: m + 46 + Math.random() * (g.h - m * 2 - 100),
          r: 34 * g.targetScale,
          born: g.t
        });
      }
      s.targets = s.targets.filter(function (t) { return g.t - t.born < s.life; });
      g.hudSub = s.hits + ' hits';
    },

    draw: function (g, ctx) {
      var s = g.state;
      for (var i = 0; i < s.targets.length; i++) {
        var t = s.targets[i];
        var age = (g.t - t.born) / s.life;
        drawTarget(ctx, t.x, t.y, t.r * (1 - age * 0.35), { decay: 1 - age });
      }
    },

    onTap: function (g, x, y) {
      var s = g.state;
      s.taps++;
      for (var i = 0; i < s.targets.length; i++) {
        var t = s.targets[i];
        var d = Math.hypot(t.x - x, t.y - y);
        if (d <= t.r) {
          s.hits++;
          s.offsets.push(d);
          g.hit({ x: t.x, y: t.y });
          s.targets.splice(i, 1);
          return;
        }
      }
      g.missFx();
    },

    finish: function (g) {
      var s = g.state;
      var perMin = s.hits / Math.max(1, g.elapsed) * 60;
      return {
        primary: perMin,
        counted: s.hits >= 5,
        metrics: [
          { k: 'Hits', v: String(s.hits) },
          { k: 'Accuracy', v: Math.round(pct(s.hits, s.taps)) + '%' },
          { k: 'Mean offset', v: s.offsets.length ? Math.round(median(s.offsets)) + 'px' : '—' },
          { k: 'Centre hits', v: String(s.offsets.filter(function (d) { return d < 12; }).length) }
        ],
        acc: pct(s.hits, s.taps)
      };
    },

    coach: function (r) {
      if (r.acc < 65) return 'A third of your taps landed on empty screen. Slow down slightly — a missed tap costs more time than a deliberate one.';
      if (r.primary > 55) return 'Fast and clean. Drop to Small targets in Setup to keep improving.';
      return 'Good rate. Try to cut your median offset down — landing nearer the centre of each target is what carries over to real fire buttons.';
    }
  });

  /* ---- 6. Reaction ---- */
  list.push({
    id: 'reaction',
    name: 'Reaction Time',
    icon: 'reaction',
    blurb: 'Tap the instant the screen lights up.',
    trains: 'Pure trigger latency. It is the floor under every other number in this app and it responds well to sleep and warm-up.',
    how: [
      'Wait with your thumb ready over the screen.',
      'The moment the field flashes orange, tap anywhere.',
      'Tapping early is a false start and repeats the round.',
      'Six rounds, scored on the median.'
    ],
    mode: 'tap',
    untimed: true,
    duration: 0,
    primary: { label: 'median reaction', unit: 'ms', decimals: 0, lowerIsBetter: true },

    init: function (g) {
      g.state = { round: 0, total: 6, waitUntil: g.t + 1.2 + Math.random() * 2.2, live: false, times: [], falseStarts: 0 };
      g.hudSub = 'round 1 of 6';
    },

    update: function (g, dt) {
      var s = g.state;
      if (!s.live && g.t >= s.waitUntil) {
        s.live = true;
        s.liveAt = g.t;
      }
    },

    draw: function (g, ctx) {
      var s = g.state;
      ctx.save();
      if (s.live) {
        ctx.fillStyle = 'rgba(255,107,53,.9)';
        ctx.fillRect(0, 0, g.w, g.h);
        ctx.fillStyle = '#1a0d05';
        ctx.font = '700 26px -apple-system,system-ui,sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('TAP', g.cx, g.cy);
      } else {
        ctx.fillStyle = '#8b9bb0';
        ctx.font = '600 17px -apple-system,system-ui,sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('wait for orange', g.cx, g.cy);
      }
      ctx.restore();
    },

    onTap: function (g) {
      var s = g.state;
      if (!s.live) {
        s.falseStarts++;
        s.waitUntil = g.t + 1.2 + Math.random() * 2.2;
        g.missFx();
        return;
      }
      s.times.push((g.t - s.liveAt) * 1000);
      s.live = false;
      s.round++;
      g.hit({ x: g.cx, y: g.cy });
      if (s.round >= s.total) {
        g.finishNow();
        return;
      }
      s.waitUntil = g.t + 1.2 + Math.random() * 2.2;
      g.hudSub = 'round ' + (s.round + 1) + ' of ' + s.total;
    },

    finish: function (g) {
      var s = g.state;
      return {
        primary: s.times.length ? median(s.times) : 0,
        counted: s.times.length >= 4,
        metrics: [
          { k: 'Median', v: s.times.length ? Math.round(median(s.times)) + 'ms' : '—' },
          { k: 'Fastest', v: s.times.length ? Math.round(Math.min.apply(null, s.times)) + 'ms' : '—' },
          { k: 'Slowest', v: s.times.length ? Math.round(Math.max.apply(null, s.times)) + 'ms' : '—' },
          { k: 'False starts', v: String(s.falseStarts) }
        ],
        falseStarts: s.falseStarts
      };
    },

    coach: function (r) {
      if (r.falseStarts > 2) return 'Several false starts. You are guessing the timing rather than reacting to it. Let the colour trigger the tap.';
      if (r.primary < 250) return 'That is a genuinely fast trigger. Your losses are not reaction time — put your practice into Tracking and Recoil Control.';
      if (r.primary > 400) return 'Slow for a warm thumb. Run this again after a couple of minutes of Precision Taps; cold hands cost 50 to 80 milliseconds.';
      return 'Normal range. Reaction time moves slowly, so treat this as a warm-up check rather than a daily target.';
    }
  });

  var ICONS = {
    flick: '<path d="M3 21 21 3M21 3h-7M21 3v7" stroke="#ff6b35" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    track: '<circle cx="12" cy="12" r="7" stroke="#ff6b35" stroke-width="2" fill="none"/><path d="M2 12h3M19 12h3" stroke="#ffb03a" stroke-width="2" stroke-linecap="round"/>',
    switch: '<circle cx="7" cy="7" r="3.5" stroke="#8b9bb0" stroke-width="2" fill="none"/><circle cx="17" cy="17" r="3.5" stroke="#ff6b35" stroke-width="2" fill="none"/><path d="M10 10l4 4" stroke="#ffb03a" stroke-width="2" stroke-linecap="round"/>',
    recoil: '<path d="M12 21V5M12 3l5 5M12 3 7 8" stroke="#ff6b35" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    taps: '<circle cx="12" cy="12" r="8" stroke="#ff6b35" stroke-width="2" fill="none"/><circle cx="12" cy="12" r="2.5" fill="#ffb03a"/>',
    reaction: '<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z" fill="#ffb03a"/>'
  };

  AT.drills = {
    all: list,
    icon: function (name) { return ICONS[name] || ICONS.taps; },
    byId: function (id) {
      for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
      return null;
    }
  };

  window.AT = AT;
})();
